# Datasets — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED`
· `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

Every figure in this document was **measured from the data on disk**, not copied from a paper or a
dataset schema. Where a corpus was only partially acquired, or where the local subset differs from the
official release, that is stated — not smoothed over. Where a fact is not established from the
available evidence, this document writes
`UNKNOWN — not established from the available evidence`.

**The corpora this project trains and evaluates on:**

| Task | Dataset | Role | Local state |
|---|---|---|---|
| `change` | **LEVIR-CD-256** | train / val / test | ACQUIRED — split 7120 / 1024 / 2048 |
| `change_vqa` | **CDVQA** (annotations) + **SECOND** (imagery) | train / val / test / test2 | annotations + imagery ACQUIRED; name-verified 2,968 / 2,968 |
| `grounding` | **VRSBench** | train / eval (16,159 eval records) | ACQUIRED — boxes normalised 0–100 |
| `optical_sar` | **BigEarthNet** (reBEN v2, CLC-19) | train / held-out test | PARTIAL — 28,000-patch local selection |
| `vlm` | **BigEarthNet-S2 single-label** | LoRA adaptation | PARTIAL — same 28,000-patch request, 24,732 matched |

> **Two different BigEarthNet corpora are in play and must not be conflated.** The optical-SAR fusion
> head consumes the **reBEN v2** corpus (`data/bigearthnet_v2/reben/`); the VLM LoRA adapter consumes a
> **BigEarthNet-S2 single-label** subset (`run_manifest.json` → `config.corpus_root`). They share the
> 28,000-patch request but are different trees with different label sources. §6 covers both.

**Sibling documents:** [`TRAINING.md`](TRAINING.md) (how each corpus is consumed),
[`EVALUATION.md`](EVALUATION.md) (protocols and honesty rules),
[`BENCHMARKS.md`](BENCHMARKS.md) (the numbers), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
(reproduction).

---

## Table of contents

1. [How these figures were obtained](#1-how-these-figures-were-obtained)
2. [Summary](#2-summary)
3. [LEVIR-CD-256 — change detection](#3-levir-cd-256--change-detection)
4. [CDVQA + SECOND — change-VQA](#4-cdvqa--second--change-vqa)
5. [VRSBench — grounding](#5-vrsbench--grounding)
6. [BigEarthNet — optical-SAR fusion and VLM adaptation](#6-bigearthnet--optical-sar-fusion-and-vlm-adaptation)
7. [Data hygiene and leakage controls](#7-data-hygiene-and-leakage-controls)
8. [What is NOT available — exhaustive](#8-what-is-not-available--exhaustive)
9. [What is NOT RUN / OPEN / BLOCKED for this topic](#9-what-is-not-run--open--blocked-for-this-topic)
10. [Where the evidence lives](#10-where-the-evidence-lives)

---

## 1. How these figures were obtained

Three rules govern this document.

1. **Measured, not quoted.** Every count, fraction and split size was produced by reading the files on
   disk — annotation JSON, label PNGs, a metadata parquet, or a directory listing. Where a paper's
   figure is mentioned, it is mentioned as *corroborated by* the measurement, never as the source.
2. **Discrepancies are reported, not reconciled.** Where the data contradicts the original plan, or
   where two documents disagree, both are recorded. Silently resolving a discrepancy is how a
   preprocessing change moves the frozen config hash
   (`configs/base.yaml` → `78f1e3700da15aa1`) without anyone noticing.
3. **A shortfall is a number, not a shrinkage.** When a corpus is only partially acquired, the
   shortfall is surfaced in a run manifest (e.g. `coverage.n_unmatched`), not absorbed by dropping
   examples quietly.

The measurement scripts are named inline throughout. The verifier that re-checks the release's headline
numbers is described in [`BENCHMARKS.md`](BENCHMARKS.md) §8.

---

## 2. Summary

| Corpus | Task | Local size | Split | Local state |
|---|---|---|---|---|
| LEVIR-CD-256 | `change` | 10,192 tiles (A / B / label) | 7120 / 1024 / 2048 | ACQUIRED |
| CDVQA annotations | `change_vqa` | 12 JSON, 55.0 MB | Train/Val/Test/Test2 | ACQUIRED |
| SECOND imagery | `change_vqa` | 2,968 pairs → 11,872 PNGs | n/a (paired) | ACQUIRED, name-verified 2,968/2,968 |
| VRSBench | `grounding` | 16,159 eval records; 142,390 train conversations | eval / train | ACQUIRED |
| BigEarthNet reBEN v2 | `optical_sar` | 28,000-patch selection of a 549,488-row corpus | 20,000 / 4,000 / 4,000 | PARTIAL |
| BigEarthNet-S2 single-label | `vlm` | 24,732 matched of 28,000 requested | 17,471 / 3,375 / 3,886 patches | PARTIAL |

---

## 3. LEVIR-CD-256 — change detection

The standard change-detection benchmark, 256 × 256 tiles. It is the corpus behind the release's only
`VERIFIED` headline (pooled IoU 0.8122).

### 3.1 The layout

The local corpus uses a **flat** layout (`docs/PHASE9_REAL_DATA_VERIFICATION.md`):

```
<root>/{A,B,label}/<split>_<scene>_<tile>.png
```

`A` is the pre-change image, `B` the post-change image, `label` the change mask. The loader recognises
both this flat layout and a nested layout (`training/change/dataset.py` → `LAYOUT_NESTED` /
`LAYOUT_FLAT`), and the verification recorded `layout: flat`.

| Directory | PNGs |
|---|---|
| `A` | 10,192 |
| `B` | 10,192 |
| `label` | 10,192 |

(`artifacts/change/levir_real_data_verification.json` → `dir_counts`; the check "A/B/label counts
agree" passed with detail `10192 / 10192 / 10192`.)

### 3.2 The split

The split is a **frozen invariant**, not a choice made per run. `docs/PHASE9_FREEZE.md` §3 records it:

> Scene split: **445 / 64 / 128 scenes = 7120 / 1024 / 2048 tiles**, enforced by `change.levir_split` in
> `configs/base.yaml`.

| Split | Scenes | Tiles |
|---|---|---|
| train | 445 | **7,120** |
| val | 64 | **1,024** |
| test | **128** | **2,048** |

(`artifacts/change/levir_real_data_verification.json` → `splits`.) The scene counts are the leakage
boundary; the tile counts are what the trainer sees. `training/change/dataset.py` defines
`LEVIR_SPLITS = {"train": 445, "val": 64, "test": 128}` and provides `assert_image_disjoint`.

**The split is scene-disjoint, and the disjointness is measured, not assumed.** The verification
recorded three separate checks — `train/val scene-disjoint`, `train/test scene-disjoint`,
`val/test scene-disjoint` — each passing with an empty offender list
(`artifacts/change/levir_real_data_verification.json` → `checks`, and → `scene_overlap` with all three
counts `0`).

**The declared split agrees with the derived split.** The verification compared the release's declared
split lists against a split derived from the file names:

| Split | declared | derived | agreement | declared-only | derived-only |
|---|---|---|---|---|---|
| train | 7,120 | 7,120 | 7,120 | 0 | 0 |
| val | 1,024 | 1,024 | 1,024 | 0 | 0 |
| test | 2,048 | 2,048 | 2,048 | 0 | 0 |

(`artifacts/change/levir_real_data_verification.json` → `split_list_agreement`.) A disagreement here
would mean the split lists and the file names told different stories; they do not.

### 3.3 The label contract

| Property | Value | Source |
|---|---|---|
| label values | `0`, `156`, `255` | `artifacts/change/levir_real_data_verification.json` → `sample.label_values` |
| `LABEL_CHANGE_LEVEL` | **128** | `training/change/train.py` |
| binary decision | `>= 128` → change | same |
| label is binary `0/255` | intermediate `156` is negligible | verification check |

Measured label-value counts over a 600-tile sample
(`artifacts/change/levir_real_data_verification.json` → `sample.label_values`):

| Value | Pixels |
|---|---|
| 0 | 37,590,438 |
| 156 | **17** |
| 255 | 1,731,145 |

The `156` pixels are **4.32e-07 of 39,321,600 px** — a negligible anti-aliasing residue. The check
"labels binary 0/255 (intermediate negligible)" passed, and the check "intermediate labels >=
LABEL_CHANGE_LEVEL (128)" passed with an empty `below-threshold` set. The threshold matters: it is what
makes the `156` residue fall on the change side rather than being silently rounded away.

Two further checks passed: "T1 and T2 are not identical" (0 identical pairs in the sample — a pair that
was byte-identical would be a null change example masquerading as a real one) and "label matches image
size" (0 mismatches).

### 3.4 The test split, measured in detail

`artifacts/change/eval_test/eval_result.json`, `split: test`:

| Property | Value |
|---|---|
| n | **2,048** |
| images containing change | **935** |
| **mean change fraction** | **0.0509** (≈ 5 % of pixels) |
| change-fraction quantiles | min 0.0, p50 0.0, p90 0.197205, max 0.684937 |
| threshold | 0.50 |
| tile size | 256 |
| device (eval) | cuda |
| total pixels scored | 134,217,728 |
| eval seconds | 55.359 |
| `config_drift` | false |

**Per-pixel confusion counts** (pooled; identical for the macro block, which aggregates the same pixels
per image):

| Count | Value |
|---|---|
| tp | 5,978,997 |
| fp | 523,658 |
| fn | 858,407 |
| tn | 126,856,666 |

The confusion counts are stored **so any metric can be recomputed rather than trusted**. The
per-image change fraction is stored for all 2,048 tiles
(`per_image_change_fraction`, a 2,048-element list), which is what makes the quantiles above
recomputable.

### 3.5 Why both pooled and macro are reported

With a **~5 % mean change fraction** the class imbalance is severe, so the pooled and macro metrics
answer different questions and both are reported
([`BENCHMARKS.md`](BENCHMARKS.md) §10.1):

| Metric | pooled | macro |
|---|---|---|
| f1 | **0.8964** | 0.7962 |
| iou | **0.8122** | 0.7180 |
| miou | 0.9007 | **0.8457** |
| precision | 0.9195 | 0.8506 |
| recall | 0.8745 | 0.7757 |

**Naming caveat.** The `macro` block carries both `iou` (0.7180, the macro-average of the per-image IoU)
and `miou` (0.8457, the mean IoU). The headline "macro IoU 0.8457" used across the release docs is the
**`miou`** field. Both key paths are named here so the two cannot be conflated.

**A threshold sweep was performed and the lever was CLOSED.** It was **val-only by construction**:
`scripts/sweep_change_threshold.py` refuses any `--split` other than `val` and exits `4` before loading
config, checkpoint, or data; its artifact records `test_split_touched: false` and its sha256 is
`34e20f62bc1dd7810f5ef5f26838213d59368cf1352a804d38cb0e140eaf97f2`. The best available gain was
**0.0007** pooled IoU against a **0.0019** epoch-to-epoch spread — 0.37× the noise. **0.50 is retained;
this hypothesis is eliminated, not deferred** (`docs/PHASE9_FREEZE.md` §5).

### 3.6 The three real-data defects found and fixed

`docs/PHASE9_REAL_DATA_VERIFICATION.md` records three defects that the verification caught before any
number was trusted. They are included here because each is a *silent* failure mode a reader could
re-introduce:

| # | Defect | Consequence if unfixed |
|---|---|---|
| 1 | **loader returned 0 items** | a training run on an empty dataset |
| 2 | **vacuous leakage guard** | a scene-disjointness check that could not fail, because it compared nothing |
| 3 | **the test split was folded into training** | a test number that was partly a training number |

Defect 3 is the dangerous one: it is exactly the failure the scene-disjointness rule exists to prevent,
and it would have *raised* the reported test IoU rather than lowering it. The verification's three
`scene-disjoint` checks (§3.2) are the pin that now prevents it.

### 3.7 The registration gate — an `OPEN` serving-path defect

The change specialist's registration gate produces false positives: **1,202 / 2,048 test tiles
(58.7 %)** flagged as mis-registered, with the flag rate climbing 42 % → 75 % → 98 % → 100 % as
ground-truth change fraction rises; **5,679 region claims suppressed** on 638 flagged tiles that do
contain change. An independent NCC check found **0 / 30** credible large offsets on the tiles the gate
blames. **`OPEN`** (`docs/PHASE9_FREEZE.md` §7). This affects the *serving path*, not the 0.8122
benchmark, which is computed offline from the mask directly.

---

## 4. CDVQA + SECOND — change-VQA

### 4.1 The two-corpus arrangement

**The CDVQA repository publishes annotations only — no imagery.** The imagery is publicly available as
**SECOND** (semantic change detection). The local corpus is therefore assembled from two sources:

```
data/cdvqa/
  annotations/   12 JSON files (55.0 MB)          <- CDVQA repo (annotations only)
  im1/ im2/      pre / post imagery   2,968 PNGs each   <- SECOND (archive order)
  label1/ label2/ pre / post label maps 2,968 PNGs each <- SECOND
  second.zip     2,406,111,691 B (kept, not deleted)   <- the raw SECOND archive
```

Source of the annotations: `https://github.com/YZHJessica/CDVQA`.
Source of the imagery: the **SECOND** semantic-change-detection dataset, downloaded and extracted
locally. The paper states the arrangement verbatim (`docs/PHASE10_CDVQA_DATA_STATUS.md` §6):

> *"we choose the existing semantic change detection dataset SECOND as the basic data to automatically
> generate a CDVQA dataset."* … *"Thus, we use the 2,968 openly available pairs as our basic data."*

### 4.2 SECOND — acquisition and verification

The imagery acquisition is a story with three findings, all recorded rather than hidden
(`docs/PHASE10_CDVQA_DATA_STATUS.md` §6).

**a. The archive is RAR, not zip.** It is *named* `second.zip` but the magic bytes are
`Rar!\x1a\x07\x01\x00` — a **RAR5** archive. The overlap checker originally assumed zip and reported
*"not a readable zip"* on a perfectly good download. It now detects the format by magic bytes and reads
RAR via libarchive. **Do not trust a `.zip` extension.**

**b. The names match — 2,968 / 2,968.** CDVQA's `file_name` values are a **sparse global identifier**,
not a compact index: min `00003`, max `24649`, **not contiguous** (21,679 gaps in a 24,647-wide span).
Whether SECOND used the same filenames was not established and had to be settled by download, not
assumption. It was:

```
SECOND PNG files   : 2,968
CDVQA unique names : 2,968   (expected 2,968)
intersection       : 2,968
CDVQA not in SECOND: 0
SECOND not in CDVQA: 0
VERDICT: MATCH
```

(`artifacts/cdvqa/second_overlap.json`: `verdict: "MATCH"`, `intersection: 2968`,
`cdvqa_not_in_second: 0`, `second_not_in_cdvqa: 0`, `cdvqa_count_matches_expectation: true`.) The checker
`scripts/check_cdvqa_second_overlap.py` exited **0**.

| SECOND archive property | Value |
|---|---|
| bytes | **2,406,111,691** |
| sha256 | `5ee2a82b5824b3f5e3c5bfaf018835862e7623993ff9ac5ce57fd11fcfad2b4e` |
| container | **RAR5** (magic `Rar!\x1a\x07\x01\x00`) |
| image format | 512 × 512, 8-bit, **PNG colour type 2** (truecolour) |
| pairs | 2,968 → 5,936 imagery PNGs + 5,936 label PNGs |

**c. The archive is a temporal PAIR.** SECOND's layout is `im1/ im2/ label1/ label2/`, and `im1` and
`im2` **share basenames** (`im1/00003.png` and `im2/00003.png` both exist). So one CDVQA `file_name`
denotes a **pair**, not an image. This is why the adapter is pair-aware and why
`imagery_verification.json` includes the check *"flat image_path() does NOT resolve (pair layout is what
is on disk)"* — the flat layout is a real, loadable-looking layout that is wrong for this corpus.

The extraction is verified end to end (`artifacts/cdvqa/imagery_verification.json`,
`all_checks_passed: true`): each of the four archive dirs holds 2,968 PNGs; `load_cdvqa(
require_images=True)` succeeds; every declared name exists in `im1`, `im2` and both label dirs; all
**153,130** examples' `image_path` **and** `image_paths` entries resolve on disk (153,130 / 153,130);
the layout is `second_pairs`; and `summarize` reports 2,968 declared / 2,968 present.

### 4.3 The annotations, measured

| File | Bytes |
|---|---|
| `Train_images.json` | 3,358,035 |
| `Train_questions.json` | 13,201,947 |
| `Train_answers.json` | 6,931,041 |
| `Val_images.json` | 826,107 |
| `Val_questions.json` | 3,259,814 |
| `Val_answers.json` | 1,710,345 |
| `Test_images.json` | 2,021,288 |
| `Test_questions.json` | 7,921,042 |
| `Test_answers.json` | 4,160,913 |
| `Test2_images.json` | 1,961,840 |
| `Test2_questions.json` | 6,353,771 |
| `Test2_answers.json` | 3,278,618 |

**Total 55.0 MB** (`docs/PHASE10_CDVQA_DATA_STATUS.md` §1).

**Schema.** Each JSON has exactly one top-level key — `images`, `questions`, `answers`:

```
images    {id, res_x, res_y, questions_ids, file_name, active}
questions {id, date_added, img_id, type, question, answers_ids, active}
answers   {id, date_added, question_id, answer, active}
```

Two type traps a reader gets wrong by guessing: `date_added` is a **float unix timestamp** (e.g.
`1631214265.2322025`), and `res_x` / `res_y` are the **strings** `".1524m"` on every row of every split —
metadata, not a usable resolution figure.

**`images` has one row per question, not per image.** Train carries **25,600 image rows** but only
**1,600 distinct `file_name` values**. Deduplicating on `file_name` and unioning `questions_ids` is
mandatory; a reader that counts image rows reports a corpus **16× too large**.

### 4.4 The measured split sizes

| Split | image rows | **unique `file_name`** | questions | answers |
|---|---|---|---|---|
| Train | 25,600 | **1,600** | **65,967** | 65,967 |
| Val | 6,400 | **400** | **16,441** | 16,441 |
| Test | 15,488 | **968** | **39,686** | 39,686 |
| Test2 | 15,488 | **968** | **31,036** | 31,036 |
| **union** | | **2,968** | **153,130** | 153,130 |

This independently confirms the paper's 1,600 / 400 / 968 image pairs and 65,967 / 16,441 / 39,686 /
31,036 question-answer pairs (`docs/PHASE10_CDVQA_DATA_STATUS.md` §3). The same figures are asserted as
constants in `training/change_vqa/dataset.py` (`EXPECTED_SPLIT_SIZES`, `EXPECTED_TOTAL_SCENES: 2968`,
`EXPECTED_MANIFEST_RECORDS: 3936`) and re-measured by the notebook's integrity gate.

### 4.5 Finding 1 — Test and Test2 are the same 968 images

Measured cross-split `file_name` overlap, all 16 cells
(`docs/PHASE10_CDVQA_DATA_STATUS.md` §4):

| | Train | Val | Test | Test2 |
|---|---|---|---|---|
| **Train** | 1600 | 0 | 0 | 0 |
| **Val** | 0 | 400 | 0 | 0 |
| **Test** | 0 | 0 | 968 | **968** |
| **Test2** | 0 | 0 | **968** | 968 |

Train / Val / Test are **cleanly disjoint** — no leakage. But **Test and Test2 share 100 % of their
images.** They are two different question sets asked over the **same 968 scenes**, not two independent
held-out samples. The paper says this outright: *"Note that there is an overlap between the two test
sets."* The measurement confirms it exactly.

> **Consequence.** Pooling Test and Test2, or reporting them as two independent evaluations, is
> statistically wrong and inflates apparent confidence. Any metric computed over `Test ∪ Test2` is a
> single evaluation of 968 images with ~70k questions, not two evaluations of 968 each.

This is why the release quotes change-VQA under **two test sets** — Test 0.697626 / 0.378373 and Test2
0.651469 / 0.372309 — and never pools them ([`BENCHMARKS.md`](BENCHMARKS.md) §4.4,
[`EVALUATION.md`](EVALUATION.md) rule 3). The allowed shared pair is recorded in the run record:
`dataset.integrity.allowed_shared_pairs: [["Test", "Test2"]]`
(`artifacts/change_vqa/run/run_record.json`).

### 4.6 Finding 2 — `change_ratio_types` has a different vocabulary in every split

The eight question types, with total counts across all four splits and their answer vocabularies
(`docs/PHASE10_CDVQA_DATA_STATUS.md` §5):

| Question type | Questions | Answer vocabulary |
|---|---|---|
| `change_or_not` | 47,807 | 2 — `yes`, `no` |
| `change_ratio_types` | 29,576 | **7–9, split-dependent** |
| `increase_or_not` | 16,040 | 2 — `yes`, `no` |
| `decrease_or_not` | 15,986 | 2 — `yes`, `no` |
| `change_to_what` | 12,233 | 6 — change classes |
| `smallest_change` | 11,808 | 6 — change classes |
| `largest_change` | 11,808 | 6 — change classes |
| `change_ratio` | 7,872 | 11 — `0` + ten 10 % bins |

**`change_ratio_types` observed values differ per split:**

| Split | observed values |
|---|---|
| Train | 9 — `0` … `70_to_80` |
| Val | 7 — `0` … `50_to_60` (no `60_to_70`, no `70_to_80`) |
| Test | 8 — `0` … `60_to_70` (no `70_to_80`) |
| Test2 | 8 — same as Test |

Val can never produce `60_to_70`. Train can never produce `80_to_90` or `90_to_100`. **A single global
label space for this question type would score the model on bins its split does not contain**, and
per-type confusion matrices for `change_ratio_types` are **not comparable across splits**.

**`change_ratio` and `change_ratio_types` are different question types with different vocabularies**
despite the similar names — `change_ratio` has 11 values including `0`; `change_ratio_types` has a
9-value union. They must not share a constant. The remaining six types are stable across all four
splits.

**The corpus uses underscore tokens** (`0_to_10`); the paper's prose writes them as percentages
(`0%-10%`). Same bins, different rendering — do not string-match the paper against the corpus.

### 4.7 Finding 3 — the imagery is not shipped with the annotations

Covered in §4.1–§4.2. The measurement that ties it together: the union of unique `file_name` across
Train/Val/Test is **exactly 2,968**, which is exactly the number of SECOND's publicly available pairs —
consistent with the paper's statement that CDVQA is generated from SECOND's 2,968 pairs.

### 4.8 Temporal and label semantics — established, with honest uncertainty

This is the part of the corpus where the difference between **established** and **supported** matters
most, and the release keeps them apart
(`artifacts/cdvqa/temporal_order_evidence_v2.json` → `note`, verbatim):

> *"label1=pre / label2=post is established from the annotations against the FIXED SECOND palette (no
> fitting) — the reverse is directionally empty. im1=pre / im2=post is SUPPORTED by a content test
> (water darkness / vegetation greenness vs a white control), not proven. Nothing is inferred from
> directory names."*

| Mapping | State | Basis |
|---|---|---|
| **`label1` = pre, `label2` = post** | **ESTABLISHED** | the annotations, against the **fixed** SECOND palette, with **no fitting** |
| **`im1` = pre, `im2` = post** | **SUPPORTED — NOT PROVEN** | a content test (water darkness / vegetation greenness) vs a white control |

**The label-order evidence.** Two hypotheses were tested against the **fixed** SECOND palette:

| Hypothesis | overall agreement | n | classes collapsed to background |
|---|---|---|---|
| **h1 (label1 = pre, label2 = post)** | **1.0** | 18,724 | **none** |
| h2 (the reverse) | 0.7602 | 29,024 | `trees`, `water`, `playgrounds` |

Under h1 every class matches the palette with agreement **1.0** and **no class collapses**. Under h2
three classes collapse to background and agreement is 0.7602. The reverse is **directionally empty** —
this is the "established" basis.

**Why v2 supersedes v1.** The supersession reason is recorded verbatim: v1 *"fit the palette in-sample
to maximise H1 and computed the reversed rate as 1 - h1_rate, so the perfect split was guaranteed by
construction. v2 measures BOTH directions against the FIXED SECOND palette and reports directional vs
tie evidence separately."* A perfect separation produced by in-sample fitting is not evidence; v2
removes the fitting.

**A corroborating annotation signal.** `change_to_what` (a *change* question) agrees far better with an
answer derived from **label2** than from label1 — 0.9469 (n 9,104) vs 0.4079 (n 7,179) — and the gap is
stable across subsamples of 500 / 800 / 1,200 / all. A "what did it change *to*" question is answered
by the post-change map; the measurement says label2 is the post map.

**The image-order evidence — supportive, statistical, not exact.** `image_label_pairing`:

| Test | Value | n |
|---|---|---|
| `water_label1_im1_darker` (mean) | **0.826** | 500 |
| `water_label2_im2_darker` (mean) | 0.7076 | 472 |
| **white control: `im1` darker rate** | **0.4356** | 2,968 |
| `veg_label1_im1_greener` (mean) | 0.7139 | 2,415 |
| `veg_label2_im2_greener` (mean) | 0.7508 | 2,392 |

The water-mask test is the strongest: where `label1` marks water, `im1` is darker than `im2` in **82.6 %**
of cases, against a **white-pixel control rate of 43.6 %** — a ~39-point gap. The check is recorded as
`strength: "supportive, statistical (not exact)"`. **It is not a proof, and this document does not call
it one.** Nothing is inferred from directory names: `imagery_verification.json` records
`temporal_order_claimed: false`, and the adapter returns `image_paths()` in **archive order** by
construction, which does not itself establish order.

**Label palette** (fixed, from SECOND; `fixed_palette` and the `label_order` per-class RGB):

| Class | RGB |
|---|---|
| `NVG_surface` | (128, 128, 128) |
| `buildings` | (128, 0, 0) |
| `low_vegetation` | (0, 128, 0) |
| `playgrounds` | (255, 0, 0) |
| `trees` | (0, 255, 0) |
| `water` | (0, 0, 255) |

`white_semantics`: `white_share: 0.8006`, `nonwhite_where_label1_eq_label2: 0.0423` — the white pixels
are the background/unchanged class, which is why the white control is the right null for the darkness
test.

### 4.9 The corpus-root contract

**Corpus root:** `data/cdvqa` — the directory that *contains* `annotations/`.

Passing `data/cdvqa/annotations` is **rejected** by `_resolve_root`. Verified by execution
(`docs/PHASE10_CDVQA_DATA_STATUS.md`, header correction):

```
load_cdvqa('data/cdvqa/annotations')
  -> raises: no 'annotations' directory under data\cdvqa\annotations

load_cdvqa('data/cdvqa')
  -> loads (Val: 400 images, 16,441 questions)
```

The correct call is `load_cdvqa('data/cdvqa')`. This is a contract, not a preference: the loader looks
for `annotations/` **under** the root it is given.

### 4.10 The answer space and the question-type resolver

The change-VQA head answers in a **closed** space of **19** answers
(`training/change_vqa/vocab.py`; `artifacts/change_vqa/run/run_record.json` →
`dataset.statistics`):

| Group | Members |
|---|---|
| binary (2) | `yes`, `no` |
| change classes (6) | `NVG_surface`, `buildings`, `low_vegetation`, `trees`, `water`, `playgrounds` |
| ratio bins (11) | `0`, `0_to_10`, `10_to_20`, `20_to_30`, `30_to_40`, `40_to_50`, `50_to_60`, `60_to_70`, `70_to_80`, `80_to_90`, `90_to_100` |

`n_distinct_answers: 19`; `answers_outside_frozen_vocabulary: []`;
`answers_illegal_for_their_type: 0`. The measured answer distribution over the 82,408 Train+Val records
is led by `no` (25,612) and `yes` (22,396), with the rarest answers being `50_to_60` (175) and
`playgrounds` (202).

**Eight question types** map to the answer space through `resolve_question_type`
(`training/change_vqa/vocab.py`, with 8 types plus an `unknown` slot → 9 resolver slots and 3 temporal
slots). Measured counts over Train+Val:

| Question type | Count |
|---|---|
| `change_or_not` | 28,799 |
| `change_ratio_types` | 12,149 |
| `increase_or_not` | 9,635 |
| `decrease_or_not` | 9,574 |
| `change_to_what` | 6,251 |
| `largest_change` | 6,000 |
| `smallest_change` | 6,000 |
| `change_ratio` | 4,000 |

The resolver's agreement rate with the corpus's own type labels is **0.9472** (`type_resolver_agreement:
78059`, `type_resolver_agreement_rate: 0.9472`) — i.e. the resolver disagrees with the corpus on ~5 % of
questions, which is recorded rather than rounded to 1.0. `validate_vocabulary()` raises if one ratio bin
or one change class is removed from the ontology.

### 4.11 The split-integrity gate and the F1 defect

**The corpus has a per-split question-id space, and this is a measured trap.**
`question_id` restarts at 0 in every split: Train `0..65966`, Val `0..16440`, Test `0..39685`, Test2
`0..31035` — all measured, **0 gaps** (`R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §5, F1).

The original `build_cdvqa_examples` indexed its question→scene map on the **bare `question_id`**, so a
multi-split call let a later split overwrite an earlier one. Measured:

| Call | questions | scenes | correct? |
|---|---|---|---|
| `splits=("Train",)` | 65,967 | 1,600 | yes, 0 pairing errors |
| `splits=("Val",)` | 16,441 | 400 | yes, 0 pairing errors |
| `splits=("Train","Val")` | 82,408 | **1,601** | **no** — correct is 2,000 |

A mis-paired question still trained, still evaluated, and still reported a number — which is why it
survived. **Fixed at the root**: the key is now `(split, question_id)`, a **no-op for a single split**,
preserving the function signature, the 10-key output schema, the split identity and Test/Test2
semantics. It fixes Test/Test2 as well, where Test2's 31,036 ids used to clobber Test's over the whole
overlap and the wrong pairing was invisible because the two splits share their 968 scenes.

**The gate.** `training/change_vqa/dataset.py` provides `verify_split_integrity` and
`assert_split_integrity`; the latter raises when a record set has two splits sharing scenes. The
trainer's `FORBIDDEN_SPLITS` are `("Test", "Test2")`, and the notebook refuses to continue unless the
four measured split counts match exactly. The smoke test proves Test/Test2 are unreachable **the strong
way**: their annotation files are written as **invalid JSON**, so a run that completes *is* the proof
they were never parsed, and a Test-only sentinel string must not appear in the serialised run record.

### 4.12 What the raw loader does *not* do

The CDVQA adapter is pair-aware and `require_images=True` succeeds — but the **raw corpus has no
training loop of its own**. The shipped `change_vqa` head trains on **cached features**, not on the raw
loader. These are different code paths and are not conflated (`TRAINING.md` §8). The R-02 status
document states the starting point was "worse than adapter-only": `training/data/cdvqa.py` decodes no
imagery, builds no tensors, and has no reasoning path — R-02 is greenfield for the reasoning layer.

---

## 5. VRSBench — grounding

VRSBench is the grounding training and evaluation source. The schema was **VERIFIED 2026-09-16 against
the real dataset files** downloaded from the official mirror (HF `xiang709/VRSBench`, the dataset
author's own repo, 4,669 downloads, **cc-by-4.0**). Published figures, current in the repo: 29,614
images, 52,472 object references, 1,232,221 VQA pairs, human-verified captions
(`training/data/vrsbench.py`).

### 5.1 What the verified files actually contain

`VRSBench_EVAL_referring.json` — **16,159 records**:

```
{
  "image_id":     "P0003_0002.png",        # bare filename, lives in Images_val.zip
  "question":     "The large yellow vehicle situated closest to the green area.",
  "ground_truth": "{<25><40><33><60>}",    # 0-100 xyxy as a token STRING, not a list
  "obj_corner":   [x1,y1, x2,y2, x3,y3, x4,y4],  # 0-1 polygon (metadata; GT string is authority)
  "obj_cls", "dataset", "question_id", "type", "size_group"
}
```

`VRSBench_train.json` — **142,390 conversations**, with tasks interleaved
(`[caption]`, `[refer]`, `[vqa]`; first-20k census 3,089 / 4,762 / 12,149).

`n_eval_records: 16159` is recorded in `artifacts/grounding/remoteclip_grounding_v001/
eval_result_canonical.json`.

### 5.2 The box convention — 0–100 vs 0–1

**VRSBench stores boxes normalised to 0–100.** This project stores boxes normalised to **0–1**. The
conversion is declared explicitly as `VRSBENCH_BOX_SCALE = 100.0` (`training/data/vrsbench.py`) and
`grounding.benchmark_box_scale: 100.0` (`configs/base.yaml`), never an inline divide — so it cannot be
applied twice or forgotten.

The module docstring states the failure mode verbatim:

> *"Treating 0-100 values as 0-1 would produce boxes a hundred times too large — an error that would
> show up as a suspiciously high IoU, not as a crash."*

`geospatial/transform.py` provides the named conversion helpers
(`benchmark_boxes_to_normalized`, `normalized_boxes_to_benchmark`, both taking `scale`), with the same
finding recorded in their docstring: *"Finding from Phase 0: VRSBench states verbatim that 'all box
coordinates are normalized to 0-100'. Treating those numbers as pixels is a silent, catastrophic
bug — this function exists so that mistake can only be made once, explicitly."*

Because the box convention is a common source of silent error, grounding is reported under **two
protocols** (canonical and matched6) and **two decode variants** — see §5.4 and
[`BENCHMARKS.md`](BENCHMARKS.md) §4.2.

### 5.3 The train split — and its noisy ground truths

> **TRAIN GROUND TRUTHS ARE NOISY. EVAL GROUND TRUTHS ARE CLEAN.**

This is stated verbatim in `training/data/vrsbench.py`, and the numbers matter:

| Split | Records | Ground-truth box range | Out-of-range |
|---|---|---|---|
| **eval** (`VRSBench_EVAL_referring.json`) | 16,159 | all within **[0, 100]** | **0** |
| **train** (`VRSBench_train.json`) | 142,390 conversations (tasks interleaved: `[caption]` / `[refer]` / `[vqa]`; first-20k census 3,089 / 4,762 / 12,149) | **36,313 refer records examined**, values reach **−73 … 196** | **8,238 of 36,313** |

Out-of-range train boxes are **rejected** by the loader, which filters that noise rather than training
on it. The loader keeps a legacy "unknown schema → fail loudly" path: any JSON file that matches neither
the verified layout nor the older key-fallback table raises with the keys it actually saw.

### 5.4 The two protocols × two decode variants

| Protocol / variant | Value | Artifact |
|---|---|---|
| canonical, `head_threshold` | **0.2838** | `eval_result_canonical.json` |
| canonical, `head_argmax` | **0.1215** | `eval_result_canonical.json` |
| matched6, `head_threshold` (top_k 6) | **0.2566** | `eval_result_matched6.json` |
| zero-shot baseline | **0.0972** | `run_record.json` → `baseline_mean_best_iou` |

**Never quote one protocol alone.** The training-time validation number is different again: best val IoU
**0.0946**, which **did not beat** the zero-shot baseline of 0.0972 (`beats_baseline: false`). That
negative result is preserved ([`TRAINING.md`](TRAINING.md) §5.8).

---

## 6. BigEarthNet — optical-SAR fusion and VLM adaptation

BigEarthNet is used in **two** places, from **two different corpora**:

1. as the **label space and evaluation benchmark** for the optical-SAR fusion head — the **reBEN v2**
   corpus at `data/bigearthnet_v2/reben/`;
2. as the **instruction-pair source** for the SmolVLM LoRA adaptation — a **BigEarthNet-S2 single-label**
   subset.

### 6.1 The reBEN v2 corpus (optical-SAR fusion)

**The label source is the official manifest, not per-patch metadata.** The local corpus has **no
`metadata.json`** in any patch directory; every one of the 28,000 S2 patch directories holds only band
TIFFs. `training/data/bigearthnet.py`'s `_read_metadata` therefore returns `{}` and
`discover_patches(require_labels=True)` filters out **every** patch. The labels exist in the release's
official manifest, keyed by `patch_id`:

```
data/bigearthnet_v2/metadata.parquet   480,038 rows
  patch_id, labels, split, country, s1_name, s2v1_name,
  contains_seasonal_snow, contains_cloud_or_shadow
```

(`training/data/bigearthnet_labels.py`; `DEFAULT_METADATA_PARQUET =
"data/bigearthnet_v2/metadata.parquet"`; `_REQUIRED_COLUMNS = ("patch_id", "labels", "split")`.)

**Measured corpus figures** (`artifacts/phase12_selection/feasibility_report.json`):

| Quantity | Value |
|---|---|
| `metadata.parquet` rows | **480,038** |
| `metadata_for_patches_with_snow_cloud_or_shadow.parquet` rows | 69,450 |
| rows total (the two files together) | **549,488** |
| **tiles** | **54** |
| ground cells | **253,957** |
| impure cells | **0** |
| `tile_derivable_rows` | 549,488 |
| `tile_mismatch_patch_id_vs_s1_name` | 0 |
| `s2v1_name_rows_with_a_tile_token` | 0 |
| blocks total | **184** (train 63 / val 65 / test 56) |
| eligible patches by split | train 53,245 / val 22,303 / test 20,989 |
| official split patches | train 272,544 / val 139,577 / test 137,367 |
| skipped multi/zero-label by split | train 219,299 / val 117,274 / test 116,378 |
| patches-per-cell histogram | {1: 96,670, 2: 58,498, 3: 59,334, 4: 39,455} |

> **Tile-count discrepancy — flagged, not silently reconciled.** The verified corpus figure is
> **54 tiles** (`artifacts/phase12_selection/feasibility_report.json` → `corpus.tiles = 54`;
> `training/data/bigearthnet.py` → "the corpus spans 54 tiles"; `training/data/bigearthnet_blocks.py`
> → "52 of 54 tiles straddle two or more official [partitions]"). The sibling document
> [`MODELS.md`](MODELS.md) §3.3 states **"98 tiles"**, and the previous revision of this file repeated
> it. **No artifact read for this document reproduces 98.** The figure this document states is **54**,
> with its key path. The 98 figure is recorded here as a **documentation discrepancy** rather than
> quietly adopted or quietly deleted.

**The tile is a coarse leakage boundary; the block is the official unit.** The tile key is the
repository-wide primitive that `evaluation.leakage` uses everywhere else, but the official split's own
atomic unit is finer: every seasonal acquisition of a ground location `(tile, row, col)` carries the same
split, and all **253,957** ground cells are split-pure (**0 impure cells in 549,488 patches**). Those
locations then group into connected blocks that **nest inside a tile** (train frame, validation ring,
test core) — which is exactly why **52 of the 54 tiles straddle two or more partitions** (47 appear in
all three; only **2** are partition-pure). Using the tile key means `_official_split_scene_disjoint`
**rejects** the official split, and that rejection is a real measurement, not a formality
(`training/data/bigearthnet.py` → `scene_id` docstring; `docs/PHASE12_LABEL_POLICY_DECISION.md` §6).

### 6.2 The 28,000-patch selection

Training and evaluation do not use all 549,488 rows. A **28,000-patch selection** was drawn from the
single-label subset, scene-disjoint by T2 blocks
(`artifacts/phase12_selection/selection_manifest_seed10.jsonl`; the feasibility report records 20
per-seed manifests):

| Property | Value |
|---|---|
| selection manifest | `artifacts/phase12_selection/selection_manifest_seed10.jsonl` |
| manifest bytes / sha256 | 12,264,103 B / `fb4d8b4b024a17a842d351e9cfdbb3904648f1883406a5953a41449b375a7843` |
| records | **28,000** |
| T2 scene blocks | 61 |
| **tiles in the shipped selection** | **47** (12 of the 47 carry blocks in more than one split) |
| targets | train **20,000** / validation **4,000** / test **4,000** |
| scene key | `ben_<tile>:<tile-global ordinal>` (4-connected components of the (row, col) grid, per (tile, official split)) |
| selection order | `sha256('<seed>:<scene_id>')` ascending |
| label policy | `skip_ambiguous` |
| `all_seeds_reached_targets` / `all_seeds_scene_disjoint` | true / true |

The **tile key is too coarse** for this selection: `PairedSample.scene_id` defaults to `ben_<tile>`, but
12 of the 47 tiles carry blocks in more than one split, so the tile key would raise a spurious "the split
leaks" abort and kill the run. The manifest's `scene_id` (`ben_<tile>:<block ordinal>`) is therefore
passed through explicitly via `PairedSample.scene_key`
(`training/fusion/reben_adapter.py`).

### 6.3 The label policy — `require_single_label`

The optical-SAR fusion training and the 11.5 metric use **exactly-one-label patches only**. The policy
is `skip_ambiguous` (`feasibility_report.json` → `corpus.label_policy`), and
`training/fusion/extract.py` implements it as `require_single_label`, counting skipped patches in
`n_skipped_by_policy`.

The label policy was **ruled by the owner** (`docs/PHASE12_LABEL_POLICY_DECISION.md`, rulings 1–6), and
the arithmetic is measured:

| Quantity | Value | Basis |
|---|---|---|
| **Mean labels per patch** | **2.9540** | full corpus |
| Mean labels per patch | 2.9495 | clean subset |
| ***f* over the full corpus** | **0.175685** (17.5685 %) | official metadata, 549,488 rows |
| ***f* over the clean subset** | **0.179934** (17.9934 %) | official metadata, 480,038 rows |
| **Single-label patches** | **96,537** | official metadata |
| Multi-label patches | **452,951** (82.43 %) | official metadata |
| Retained by `require_single_label` | 96,537 (17.57 %) = 53,245 train / 22,303 val / 20,989 test | — |
| Discarded by `require_single_label` | 452,951 patches (82.43 %) | — |

So the corpus carries **≈ 2.95 classes per patch on average**, and only **17.57 %** of patches carry
exactly one label. The *f* ≥ 7.28 % threshold the plan set is cleared with room to spare (17.57 % vs
7.28 %), but the cost is explicit: the single-label policy **discards ~1.95 of every 2.95 labels per
patch (≈ 66 %)**.

### 6.4 The single-label caveat — the figure that governs how these metrics may be used

> ### **The local subset is 100 % single-label, against the official 1–11 multi-label scheme.**
>
> Metrics computed on this subset are therefore **not comparable** to published multi-label BigEarthNet
> numbers. Any statement of the form "BigEarthNet mAP = X" is **false** for this subset.
> (`run_manifest.json` → `corpus.coverage.single_label_fraction: 1.0`, `n_multi_label: 0`;
> `corpus.warnings`.)

The corpus-level figures the caveat is measured against: **≈ 2.95 labels per patch on average (max 11)**,
with only **17.57 %** of patches single-label (§6.3). The VLM run manifest states the consequence
verbatim:

> *"every one of the 24732 matched patches is single-label; the BigEarthNet corpus at large averages
> ~2.95 labels per patch (max 11), so this subset cannot support multi-label enumeration questions"*

### 6.5 The fusion evaluation detail

`artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`:

| Property | Value |
|---|---|
| split | test |
| n scored | **4,000** |
| classes in label space | 19 |
| classes present in the scored split | **14** — `[0,2,3,4,5,6,7,8,9,10,12,13,17,18]` |
| classes absent | **5** — `[1,11,14,15,16]` |
| **macro-F1 denominator** | **"all 19 classes (absent classes contribute 0.0)"** |
| accuracy | **0.931** |
| **macro F1** | **0.434161** |
| loss | 0.254592 |
| `is_deciding_statistic` | **false** |

The macro-F1 denominator is recorded explicitly so a reader can see that absent classes drag the macro
score down **by construction**. This is why accuracy (0.931) and macro-F1 (0.434161) must be read
together, and why the ruling is `OPEN`. Classes 5 and 6 are *present* in the scored split and still
score `0.0` (`docs/PHASE12_115_METRIC_COMPUTED.md`); the alternative denominator (14 classes) would give
macro-F1 **0.589218** instead of **0.434161**.

**The 19-class label space** is `CLC19_CLASSES`, an ordered 19-name tuple in
`training/data/bigearthnet.py`. The per-class test label distribution recorded in
`docs/PHASE12_115_METRIC_COMPUTED.md` is
`[13,0,63,39,846,4,1,6,453,246,4,0,9,18,0,0,0,34,2264]` — a strongly skewed distribution dominated by
class 18.

> **The BigEarthNet documentation — its uses, mentions, or endorsements — does not specify a percentile
> stretch.** This project nevertheless applies percentile normalisation for optical inputs (2/98) to
> match the CROMA contract. That is a **deliberate, documented choice**, not an upstream fact
> (`docs/RESEARCH_NOTES.md`).

### 6.6 The VLM subset — a different BigEarthNet corpus

The VLM LoRA adapter consumes a **BigEarthNet-S2 single-label** subset
(`run_manifest.json` → `config.corpus_root`:
`/kaggle/input/datasets/creatorballs/bigearth-net-s2-single-label/BigEarthNet-S2`), **not** the reBEN v2
fusion corpus.

| Property | Value | Key path |
|---|---|---|
| instruction families | `["presence"]` | `corpus.families` |
| n samples (questions) | 49,464 | `corpus.n_samples` |
| n requested patches | 28,000 | `corpus.coverage.n_requested` |
| n matched | **24,732** | `corpus.coverage.n_matched` |
| **coverage fraction** | **0.883286** | `corpus.coverage.coverage_fraction` |
| n unmatched | **3,268** (11.7 %) | `corpus.coverage.n_unmatched` |
| single-label fraction | **1.0** | `corpus.coverage.single_label_fraction` |
| scene blocks (train / val / test) | 1,160 / 482 / 202 | `corpus.scene_counts` |
| patches (train / val / test) | 17,471 / 3,375 / 3,886 | `corpus.split_info.patches_by_split` |
| samples (train / val / test) | 34,942 / 6,750 / 7,772 | `corpus.split_counts` |
| scene key | `ben_<tile>:<k>` | `corpus.split_info.scene_key` |
| render | RGB `[B04, B03, B02]`, per-band percentile stretch 2/98, uint8 | `corpus.render` |

**The unmatched 3,268 patches are excluded, not guessed.** `corpus.warnings`, verbatim:

> *"3268 of 28000 patches have no row in the manifest and carry NO labels; they are excluded rather than
> guessed (88.3% coverage)"*

`training/data/bigearthnet_labels.py` records the same rule: a patch with no manifest row gets **no
labels** — not a default, not a nearest-neighbour, not a guessed one. It is reported as unlabelled and
excluded, and the count is surfaced by `coverage()` so the shortfall is **a number in the run manifest
rather than a silent shrinkage of the training set**. `LabelCoverage` carries `n_requested`,
`n_matched`, `n_unmatched`, `n_multi_label`, `n_single_label` and `single_label_fraction`.

**The render is per-patch, not corpus-global** — `corpus.render.note` records that this is deliberate,
"so inference needs no training-set statistics". The split policy is
`release_partition_keyed_by_T2_blocks`, with a leakage check calling
`evaluation.leakage.assert_no_scene_overlap`, and the block reconstruction is conservative:
*"blocks are reconstructed from this corpus's patches, not the full release; that can only split a block
further, never merge across a partition."*

### 6.7 The format contradiction — reported, not silently resolved

> **PARTIALLY — reported, not silently resolved.** The BigEarthNet data format contradicts the original
> plan. This was reported rather than quietly patched, because silently changing the preprocessing would
> move the frozen config hash (`78f1e3700da15aa1`). See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

The concrete contradictions the project hit, each measured:

| Contradiction | What the plan/release implied | What the data showed |
|---|---|---|
| per-patch `metadata.json` | a per-patch label source exists | **no `metadata.json` in any of the 28,000 patch dirs**; labels live only in the parquet (`training/data/bigearthnet_labels.py`) |
| S1/S2 co-located | one patch dir holds both modalities | **separate trees with disjoint names** — the S1 and S2 name sets intersect in **0 of 28,000**; the parquet's `s1_name` is the only join key (`training/fusion/reben_adapter.py`) |
| uniform band resolution | bands share one pixel size | **mixed native resolutions** — 120×120 (10 m), 60×60 (20 m), 20×20 (60 m) inside the *same* patch; resampling onto the 10 m grid is well defined because all bands share the same geographic bounds |
| a tile-level split is geographical | "one tile does not straddle train/test" | **FALSE** — 52 of 54 tiles straddle; the official unit is the ground block (`training/data/bigearthnet.py`) |

Each contradiction is handled by an **additive** module rather than by editing the existing one:
`training/fusion/reben_adapter.py` "is ADDITIVE. It does not modify `pair_patches`, `discover_patches` or
the extraction core." This is why the frozen config hash did not move when the v2 corpus was integrated.

---

## 7. Data hygiene and leakage controls

This section covers the rules that apply to **every** corpus above, and the code that enforces them.

### 7.1 Splits are by scene (or group), never by example

The classic remote-sensing leakage bug is slicing a large scene into tiles and then splitting the
**tiles** randomly. Adjacent tiles are near-duplicates, so a random tile split leaks the test set into
training. Every split in this project is at the **scene** (or **group**) level.

| Corpus | Split key | Definition |
|---|---|---|
| LEVIR-CD | scene | the LEVIR scene name inside `<split>_<scene>_<tile>.png`; 445 / 64 / 128 scenes |
| CDVQA | scene (`file_name`) | the CDVQA scene; Train/Val/Test disjoint, Test ∩ Test2 = 968 (allowed pair) |
| VRSBench | image | `training/grounding/dataset.py` splits by image |
| BigEarthNet (fusion) | **block** | `ben_<tile>:<block ordinal>` — 4-connected components of the (row, col) grid, per (tile, official split) |
| BigEarthNet (VLM) | block | `ben_<tile>:<k>` |
| Router | **group** | template family / hard-negative family (`router/dataset.py::split_by_group`) |

### 7.2 The leakage module — five rules, enforced in code

`evaluation/leakage.py` opens with the rules and states that they are "non-negotiable and are enforced in
code, not by convention":

1. **Split by SCENE, never by sample/tile.**
2. **No scene may appear in more than one split.**
3. **Duplicate content (same sha256) is rejected.**
4. **Public test sets are immutable and off-limits to training code.**
5. **Hidden data must never influence thresholds, prompts, or routing.**

And the module's own operating principle, verbatim:

> *"Every function here either returns a clean result or raises. None of them 'warn and continue' — a
> leakage warning that scrolls past is indistinguishable from no check."*

The enforcement surface:

| Function | What it does |
|---|---|
| `assign_splits_by_scene(records, *, train_ratio, val_ratio, seed)` | assigns splits at the **scene** level, deterministically. Scene keys are **sorted first**, then shuffled with a seeded RNG, so the assignment is reproducible and independent of input order. **A scene's records all receive the same split — enforced, not hoped for.** Emits in sorted scene order (the seed controls *which* scene goes to which split, never the output order). |
| `audit_manifest(manifest, *, require_all_splits)` | full audit; **raises `LeakageError`** on a scene in >1 split or a duplicate content hash. Adds advisory warnings: not all splits populated; and `scene_key` falling back to `sample_id` (which makes scene-level isolation **vacuous** for those samples). |
| `assert_no_scene_overlap(train, test)` | hard assertion that two collections share no scene; raises with the overlap count and up to five example scene keys. |
| `PublicTestFirewall(root)` | two-layer defence (§7.4). |
| `assert_no_hidden_access(config)` | guards the hidden-evaluation contract (§7.5). |
| `deduplicate(records)` | drops duplicate-content records, returning `(kept, dropped_sample_ids)`. |
| `content_hash_file(path)` | streamed SHA-256, delegating to `preprocessing/raster.py::file_sha256`. |

### 7.3 `leakage_split_key: scene_id`

Declared in `configs/base.yaml` §`evaluation`. The `scene_key` property resolves, in order, an explicit
`scene_key`, then `scene_id`, then a geographic hash, then the source scene, then falls back to
`sample_id`. The `audit_manifest` advisory explicitly flags the fallback case, because a sample whose
`scene_key` has fallen all the way back to its `sample_id` has **no** scene-level isolation at all.

### 7.4 The public-test firewall

`evaluation/leakage.py::PublicTestFirewall` (`PUBLIC_TEST_DIRNAME = "public_test"`) blocks training-time
reads of the public test tree with **two layers**:

| Layer | Method | Behaviour |
|---|---|---|
| **path inspection** | `check_read(path)` | any read whose **resolved** path is under the public test root is refused while the firewall is armed |
| **manifest inspection** | `check_manifest(manifest)` | a manifest containing test-split records may not be handed to a training loop |

The firewall is **armed by default in training scripts** and disarmed only by the evaluation runner,
which is a separate entry point. `arm()` / `disarm()` / `.armed` are exposed so the state is inspectable.
Both checks raise `LeakageError` with the offending path or record count in the context.

### 7.5 Hidden-set discipline

`assert_no_hidden_access(config)` enforces the hidden-evaluation contract at runtime:

| Condition | Behaviour |
|---|---|
| `evaluation.hidden_data_access is not False` | **raises** `LeakageError` |
| `evaluation.official_aggregate_weights is not None` | **raises** `LeakageError` |

The second is why `configs/base.yaml` records `official_aggregate_weights: null` — inventing an aggregate
formula is prohibited ([`BENCHMARKS.md`](BENCHMARKS.md) §10.7 and
[`EVALUATION.md`](EVALUATION.md) rule 8). `immutable_public_test: true` and `hidden_data_access: false`
are the two config values that make the firewall and the hidden guard meaningful.

### 7.6 Deduplication

`deduplicate()` removes records whose content sha256 has already been seen, keeping the first
occurrence and returning the dropped sample ids. Duplicates are treated as both a leakage vector **and**
a silent train/test contamination path, so they are removed, not tolerated. The dedup key is the
**content** hash (streamed SHA-256), not the file name — two differently-named copies of the same pixels
are the same record.

### 7.7 The GeoTIFF contract

The raster input contract is `preprocessing/raster.py`, and it implements the frozen validation chain
from `docs/ARCHITECTURE_FREEZE.md` §2.6:

```
file -> dimensions -> bands -> dtype -> CRS -> transform -> bounds -> nodata
     -> modality -> temporal metadata
```

Design rules, verbatim from the module docstring:

> *"Never raise a bare exception. Every failure is a typed SatQueryError. Never silently drop geospatial
> metadata. If the source had a CRS, the returned AssetMetadata says so. A missing CRS degrades to
> non-geospatial mode; it does not abort."*

`inspect_raster(path, *, max_pixels, explicit_modality, compute_hash, sensor)` **describes a raster
without loading its pixel data** and returns `AssetMetadata` with full geospatial detail preserved. The
recorded fields include `crs`, `transform` (the first 6 affine coefficients), `bounds`, `width`,
`height`, `band_count`, `dtype`, `nodata`, `resolution`, `has_crs`, `is_georeferenced`, `driver`,
`is_tiled`.

| Guard | Behaviour |
|---|---|
| file missing / not a regular file | `RasterReadError` |
| rasterio unavailable | `RasterReadError` |
| degenerate dimensions (`<= 0`) | `RasterReadError` |
| `max_pixels` exceeded | `OversizedImageError` (**recoverable** — the caller may downscale) |
| `band_count <= 0` | `UnsupportedBandsError` |
| unreadable raster | `RasterReadError` |

**Modality inference** is a heuristic and is labelled as one: `_OPTICAL_BAND_COUNTS = {3,4,8,11,12,13}`
and `_SAR_BAND_COUNTS = {1,2}`, with an explicit label honoured first. The docstring is explicit that
these are "heuristics, not ground truth — the sensor adapter is authoritative when a sensor descriptor
is supplied".

**Georeferencing is preserved through preprocessing, never silently stripped.** `read_bands` returns
`(bands, H, W)` plus a profile that retains `crs`/`transform`/`bounds`/`nodata`; `write_raster` writes
using that profile so evidence artifacts (change maps, masks) stay georeferenced. `preprocessing/
imagery.py::load_image_array` converts a raster to a displayable uint8 `(H, W, 3)` array with a
**deterministic** 2/98 percentile stretch (over finite values only, so a nodata sentinel does not crush
the dynamic range) — and it **does not resample, crop, or reproject**, because the grounding specialist
converts normalised boxes to pixel coordinates using the **original** raster's dimensions, and a silent
resize here would put every box in the wrong place.

### 7.8 CRS comparison and pair compatibility

"Can these two rasters be compared at all?" is answerable **without touching pixel arithmetic** —
`geospatial/crs.py` is kept separate from `geospatial/transform.py` for exactly that reason.

`compare_crs(left, right)` returns a `CRSCompatibility`:

| Case | `compatible` | `identical` | reason |
|---|---|---|---|
| identical CRS | true | true | "identical CRS" |
| different but same kind (both projected or both geographic) | true | false | "reprojection required (A -> B)" |
| mixed projected/geographic | true | false | "mixed projected/geographic CRS …; reprojection required and should be verified" |
| **either side missing a CRS** | **false** | false | "one or both rasters lack a CRS; spatial comparison is unsafe" |

A missing CRS on either side is **not** compatible for spatial comparison — but it is **not fatal for
non-spatial tasks** (a VQA question about a single image does not need a CRS). `is_metric(crs)` returns
`True` only when the CRS is projected and its axis unit is metres, so `pixel_area_m2()` returns `None`
for a geographic CRS rather than guessing.

**Temporal metadata and pair compatibility.** The `AssetMetadata` carries the geospatial detail, and the
pair-compatibility question ("are these two rasters a usable change pair?") is answered by the change
specialist's preconditions rather than by a single function: the pair must share a scene, and `T1 != T2`
(the two acquisitions must differ). The LEVIR verification measures this directly — the check "T1 and T2
are not identical" passed with 0 identical pairs in a 600-tile sample
(`artifacts/change/levir_real_data_verification.json`). For CDVQA the temporal question is the harder
one and is handled by the evidence engine in §4.8, because the pair is identified by a **shared
basename** in the SECOND layout, not by embedded temporal metadata.

### 7.9 The pixel ↔ geo conversion is first-class and unit-tested

`geospatial/transform.py` makes the `coordinate_system` field real. Every spatial result the system
produces passes through here at least once, so this is the single place a normalised 0–1 box, a pixel
box, and a geographic box are converted between. The non-negotiable rules, verbatim:

> *"CRS and affine transform are preserved, never silently stripped. Conversions are explicit about
> their source and target frames. A conversion that cannot be performed raises, rather than returning a
> plausible lie."*

| Function | Contract |
|---|---|
| `normalized_to_pixel(box, width, height)` | values are **not clipped** — a box slightly outside the frame is preserved so the caller can decide whether that is a bug or a legitimate edge case |
| `pixel_to_normalized(window, width, height)` | inverse; raises on non-positive dimensions |
| `pixel_to_geo(window, transform)` | uses the affine; the geographic `miny` comes from the **bottom** row, because the y axis usually points down in pixel space |
| `geo_to_pixel(bounds, transform)` | inverts the affine; raises `CoordinateError` if the transform is not invertible |
| `to_normalized` / `to_pixel` / `to_geo` | schema-aware `Box` conversions; each requires enough context for the source frame and **raises** rather than guessing |
| `intersect` / `iou` | axis-aligned intersection and IoU, used by grounding NMS and change-region overlap |
| `benchmark_boxes_to_normalized` / `normalized_boxes_to_benchmark` | the explicit 0–100 ↔ 0–1 conversion (§5.2) |
| `affine_from_metadata(geo)` | reconstructs an `Affine` from a `GeoMetadata.transform` list of length 6 |

`PixelWindow.__post_init__` raises `CoordinateError` on an **inverted** window (`col_max < col_min` or
`row_max < row_min`), so a swapped-corner box cannot propagate.

### 7.10 The deterministic input-quality gate

`preprocessing/quality.py` exists because of finding **F5-5** (`docs/PHASE5_VLM_CONTRACT.md`): a loaded
SmolVLM-500M-Instruct, given 512×512 **uniform random noise** and a prompt that explicitly says "answer
only from what is visible" and "if the image does not contain enough information, say so plainly",
produced:

> *"A black and white photograph of a man and a woman, who appear to be in a room, with a table in front
> of them."*

That is a fluent, specific, entirely fabricated scene description. The prompt cannot prevent it — a
500M-parameter VLM will describe *something* for any input, and asking it to self-assess reliably is
asking it to do the thing it just failed at.

So the gate is **deterministic and upstream of the model**. The discriminating signal is **lag-1 spatial
autocorrelation**, not variance:

| Input | Autocorrelation |
|---|---|
| real remote-sensing imagery | 0.6 – 0.99 |
| uniform random noise | ~ 0.00 |
| a constant (blank) image | undefined; variance ~ 0 |

Variance alone cannot separate these: a flat desert scene and a flat "all-zero" tile both have
near-zero variance, but the desert has strong neighbour correlation and the zero tile does not. The gate
classifies an array as `STRUCTURED`, `FLAT`, `NOISE`, `TOO_SMALL` or `INVALID_VALUES`;
`BLOCKING_VERDICTS = {NOISE, INVALID_VALUES}` must block a VLM call. The thresholds and their measured
margins:

| Threshold | Value | Measured margin |
|---|---|---|
| `MIN_AUTOCORRELATION` | 0.10 | structured 0.952 vs noise −0.008 — a **0.96 margin** |
| `FLAT_STD_EPSILON` | 1e-6 | a constant array → `FLAT` |
| `NOISE_ENTROPY_BITS` | 7.8 | structured ramp+texture 7.581 vs noise 7.988 — a **0.22-bit gap**, thin by design |

The module is explicit that **autocorrelation carries the gate** and entropy is a redundant second
signal: *"It is kept because a future change to either metric should not silently disable the gate, but
do not rely on it alone."* `assess_image_quality` has **no tolerance** for non-finite values — a fixed
fraction would be size-dependent, so one NaN in a 64×64 array (0.99976) would pass while one NaN in a
10×10 array (0.99) would fail; the same defect must not be tolerated or rejected depending on image
dimensions. Known limitation, recorded deliberately: a float GeoTIFF whose nodata sentinel is NaN is
refused, because the alternative is computing correlation over NaN.

**Nothing here is learned, sampled, or probabilistic. Same input, same verdict.**

---

## 8. What is NOT available — exhaustive

| Corpus / resource | State |
|---|---|
| **BigEarthNet S1 + S2 full corpus (549,488 rows)** | **NOT DOWNLOADED** — only the 28,000-patch selection is materialised locally |
| **BigEarthNet multi-label (reBEN) results** | **NOT PRODUCED** — the local subset is single-label; metrics are not comparable to published numbers (§6.4) |
| **BigEarthNet-S2 full single-label corpus** | **NOT DOWNLOADED** — the VLM run matched 24,732 of 28,000 requested (§6.6) |
| **Cross-dataset generalisation sets** | **not used** |
| **Any private / hidden evaluation data** | **not accessed** (`hidden_data_access: false`) |
| **The public test split, for training** | **blocked by the firewall** (§7.4) |
| **CDVQA imagery from the CDVQA repo** | **does not exist** — the repo publishes annotations only; imagery comes from SECOND (§4.1) |
| **CDVQA label-map semantics** | **partially decoded** — the palette is fixed and the order is established (§4.8), but "semantics **not decoded**" is the state recorded for `label1/`/`label2/` at acquisition (`docs/PHASE10_CDVQA_DATA_STATUS.md` §1) |
| **VRSBench caption / VQA benchmark scoring** | **NOT RUN** — the loader verifies the schema; captioning metrics are governed by R-16, which is `OPEN` |
| **VRSBench `obj_corner` polygons as training targets** | **not used** — the GT token string is the authority; `obj_corner` is metadata |
| **A `tiles` figure for the VLM BigEarthNet-S2 subset** | `UNKNOWN — not established from the available evidence` |
| **The authoritative router parameter count (50,822 vs 51,725)** | `UNKNOWN — not established from the available evidence` ([`TRAINING.md`](TRAINING.md) §4.1) |
| **A licence file for this repository** | **does not exist** — `OPEN` |
| **Kaggle runtime for the external runs** | **not measured** — the guides give measured *local* figures and label the Kaggle figures as estimates |

---

## 9. What is NOT RUN / OPEN / BLOCKED for this topic

**NOT RUN**

- Any **multi-label** BigEarthNet evaluation (the local subset is single-label).
- Any **cross-dataset** generalisation evaluation.
- The **VRSBench captioning / VQA** benchmarks (R-16 `OPEN`).
- The **change-VQA** evaluation on any split beyond Test and Test2 (no third held-out set exists).
- The **VLM adapter on a second instruction family** — `instruction_families: ["presence"]` only.
- A **full-corpus** BigEarthNet download or extraction.

**OPEN**

- The **BigEarthNet tile-count discrepancy** (54 verified vs 98 in `MODELS.md`) — reported, not resolved
  (§6.1).
- The **BigEarthNet format contradiction** — reported, not silently resolved (§6.7).
- The **CDVQA `im1`/`im2` temporal order** — `SUPPORTED — NOT PROVEN` (§4.8).
- The **CDVQA label-map semantics** — the palette is fixed and the order is established, but full
  semantic decoding of the label maps is not claimed.
- The **optical-SAR ruling** — accuracy 0.931 / macro-F1 0.434161, `is_deciding_statistic: false`
  (§6.5).
- The **change-VQA ruling** — `metric_ruling: OPEN` on both test sets.
- The **change registration gate** — 58.7 % false positives on the serving path (§3.7).
- The **licence** — no LICENSE file exists.
- **R-03** — calibration; nothing is fitted.

**BLOCKED**

- Nothing blocks any training or evaluation run. The one recorded external blocker that *was* crossed is
  the optical-SAR Arm-B feature cache (a ~2.7 h CPU extraction), which was completed
  ([`TRAINING.md`](TRAINING.md) §7.4).

---

## 10. Where the evidence lives

| Corpus | Artifact(s) |
|---|---|
| LEVIR-CD-256 | `artifacts/change/levir_real_data_verification.json`, `artifacts/change/eval_test/eval_result.json`, `artifacts/change/levir_change_v001/{run_record,training_metadata}.json`, `artifacts/change/threshold_sweep_val.json`, `training/change/dataset.py`, `training/change/train.py`, `docs/PHASE9_REAL_DATA_VERIFICATION.md`, `docs/PHASE9_FREEZE.md` |
| CDVQA + SECOND | `artifacts/cdvqa/{second_overlap,imagery_verification,temporal_order_evidence_v2}.json`, `docs/PHASE10_CDVQA_DATA_STATUS.md`, `docs/ARCHITECTURE_CHANGE_CDVQA_IMAGE_PAIRS.md`, `docs/PHASE10_ENTRY_GATE.md`, `artifacts/change_vqa/run/run_record.json`, `training/change_vqa/{dataset,vocab}.py`, `training/data/cdvqa.py` |
| VRSBench | `training/data/vrsbench.py`, `artifacts/grounding/remoteclip_grounding_v001/{run_record,training_metadata,eval_result_canonical,eval_result_matched6}.json` |
| BigEarthNet (fusion) | `artifacts/phase12_selection/feasibility_report.json`, `artifacts/phase12_selection/selection_manifest_seed10.jsonl`, `artifacts/optical_sar/fusion_head_production_v001/{pre_registered_115_metric,production_head_record}.json`, `training/data/bigearthnet.py`, `training/data/bigearthnet_labels.py`, `training/data/bigearthnet_blocks.py`, `training/fusion/reben_adapter.py`, `docs/PHASE12_LABEL_POLICY_DECISION.md`, `docs/PHASE12_115_METRIC_COMPUTED.md`, `docs/PHASE12_ENTRY_GATE.md` |
| BigEarthNet (VLM) | `.scratch/phase6_real_adapter/phase6_adapter/run_manifest.json`, `scripts/prepare_bigearthnet.py` |
| Leakage / firewall | `evaluation/leakage.py`, `evaluation/manifest_freeze.json`, `evaluation/public_test/`, `tests/leakage/` |
| GeoTIFF contract / pair compatibility | `preprocessing/raster.py`, `preprocessing/imagery.py`, `preprocessing/quality.py`, `geospatial/crs.py`, `geospatial/transform.py`, `docs/ARCHITECTURE_FREEZE.md` §2.6 |
| Quality gate | `preprocessing/quality.py`, `docs/PHASE5_VLM_CONTRACT.md` (finding F5-5) |

**Sibling documents:** [`TRAINING.md`](TRAINING.md) (how each corpus is consumed),
[`EVALUATION.md`](EVALUATION.md) (protocols), [`BENCHMARKS.md`](BENCHMARKS.md) (the numbers),
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (reproduction),
[`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) (the format contradiction),
[`LIMITATIONS.md`](LIMITATIONS.md) (the honest counterweight).
