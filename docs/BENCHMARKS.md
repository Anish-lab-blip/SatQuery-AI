# Benchmarks — deep reference

**Every number here is artifact-backed.** Where a metric exists, its source file and the **exact key
path** inside that file are named. Where a benchmark does **not** exist, that is stated explicitly and
tagged `NOT RUN` — an absent number is never silently omitted, never estimated, and never replaced with
a plausible-looking substitute.

**Status tags:** `VERIFIED` · `MEASURED` · `NOT RUN` · `OPEN` · `REJECTED` · `BLOCKED`.

**The single most important rule in this document: do not fabricate.** If a value is not in an
artifact that was read, it does not appear here. Where a fact is not established, this document writes
`UNKNOWN — not established from the available evidence`.

> **There is no system-level end-to-end benchmark.** The router → specialist → envelope pipeline has
> never been scored end-to-end, and no end-to-end accuracy is claimed anywhere in this repository. What
> exists is (a) per-specialist metrics on their own training-family test splits, and (b) a behavioural
> live-validation record (§5) that proves the pipeline *runs and routes*, not that it is *accurate*.

---

## Table of contents

1. [Headline table](#1-headline-table)
2. [Source artifacts](#2-source-artifacts)
3. [The honesty rules this document follows](#3-the-honesty-rules-this-document-follows)
4. [Per-task evaluation protocols](#4-per-task-evaluation-protocols)
   - 4.1 [Change detection](#41-change-detection--verified)
   - 4.2 [Grounding](#42-grounding--measured-two-protocols--two-decode-variants)
   - 4.3 [Optical-SAR fusion](#43-optical-sar-fusion--measured-ruling-open)
   - 4.4 [Change-VQA](#44-change-vqa--measured-ruling-open)
   - 4.5 [VLM adapter](#45-vlm-adapter--measured-acceptance-rejected)
   - 4.6 [Router](#46-router--measured-test-not-run)
   - 4.7 [Calibration](#47-calibration--measured-negative-result)
5. [Live validation — behavioural, not accuracy](#5-live-validation--behavioural-not-accuracy)
6. [What is NOT benchmarked](#6-what-is-not-benchmarked)
7. [How to reproduce](#7-how-to-reproduce)
8. [The verification tool — exactly what it checks](#8-the-verification-tool--exactly-what-it-checks)
9. [Artifact schemas and key-path resolution](#9-artifact-schemas-and-key-path-resolution)
10. [Metric glossary and aggregation arithmetic](#10-metric-glossary-and-aggregation-arithmetic)
11. [Evidence index](#11-evidence-index)

---

## 1. Headline table

Every row names its source artifact and the exact key path within it. The `n` and `split` columns are
part of the claim, not decoration: a metric without its population is not a result.

| Capability | Metric | Value | Split / protocol | n | Source artifact → key path | Status |
|---|---|---|---|---|---|---|
| Change detection | pooled IoU | **0.8122** | LEVIR-CD-256 test, thr 0.50 | 2,048 | `artifacts/change/eval_test/eval_result.json` → `metrics.pooled.iou` | **VERIFIED** |
| Change detection | macro IoU | **0.8457** | same | 2,048 | `…/eval_result.json` → `metrics.macro.miou` | **VERIFIED** |
| Change detection | pooled F1 | **0.8964** | same | 2,048 | `…/eval_result.json` → `metrics.pooled.f1` | **VERIFIED** |
| Grounding | mean best IoU | **0.2838** | VRSBench, canonical (head threshold, top_k 20) | 16,159 | `…/eval_result_canonical.json` → `results.head_threshold.mean_best_iou` | MEASURED (2 protocols) |
| Grounding | recall@0.5 | **0.2198** | canonical | 16,159 | `…/eval_result_canonical.json` → `results.head_threshold.recall.0.50` | MEASURED (2 protocols) |
| Grounding | mean best IoU | **0.2566** | VRSBench, matched6 (head threshold, top_k 6) | 16,159 | `…/eval_result_matched6.json` → `results.head_threshold.mean_best_iou` | MEASURED (2 protocols) |
| Grounding | recall@0.5 | **0.1938** | matched6 | 16,159 | `…/eval_result_matched6.json` → `results.head_threshold.recall.0.50` | MEASURED (2 protocols) |
| Grounding | head-argmax IoU | **0.1215** | canonical (argmax decode) | 16,159 | `…/eval_result_canonical.json` → `results.head_argmax.mean_best_iou` | MEASURED |
| Grounding | zero-shot baseline IoU | **0.0972** | canonical (no trained head) | 16,159 | `…/eval_result_canonical.json` → `results.zero_shot_matched.mean_best_iou` | MEASURED (baseline) |
| Optical-SAR fusion | accuracy | **0.931** | held-out test, 19 classes | 4,000 | `…/fusion_head_production_v001/pre_registered_115_metric.json` → `accuracy` | MEASURED, ruling **OPEN** |
| Optical-SAR fusion | macro F1 | **0.434161** | same | 4,000 | `…/pre_registered_115_metric.json` → `macro_f1` | MEASURED, ruling **OPEN** |
| Change-VQA | accuracy | **0.697626** | `test` | 39,686 | `artifacts/change_vqa/run/PROMOTION.json` → `verification.test_accuracy` | MEASURED, ruling **OPEN** |
| Change-VQA | macro F1 | **0.378373** | `test` | 39,686 | `…/PROMOTION.json` → `verification.test_macro_f1` | MEASURED, ruling **OPEN** |
| Change-VQA | accuracy (2nd set) | **0.651469** | `test2` | 31,036 | `…/PROMOTION.json` → `verification.test2_accuracy` | MEASURED, ruling **OPEN** |
| Change-VQA | macro F1 (2nd set) | **0.372309** | `test2` | 31,036 | `…/PROMOTION.json` → `verification.test2_macro_f1` | MEASURED, ruling **OPEN** |
| VLM (adapted) | exact_match | **0.963** | frozen 1,000-question subset | 1,000 | `artifacts/vlm/phase6_closure.json` → `why_usable_verified.adapted_test.exact_match` | MEASURED, **ACCEPTANCE-REJECTED** |
| VLM (adapted) | F1 | **0.96432** | same | 1,000 | `…/phase6_closure.json` → `why_usable_verified.adapted_test.f1` | MEASURED, **ACCEPTANCE-REJECTED** |
| Router | overall **ungated** accuracy | **0.965116** | val, corpus-limited | 86 | `artifacts/router/threshold_sweep_val.json` → `overall_ungated_accuracy` | MEASURED — **TEST NOT RUN** |
| Calibration | ECE before / after | **0.013755 → 0.014929** | val, T = 0.9773 | 16,441 | `artifacts/calibration_v001.json` → `metrics.ece_before` / `metrics.ece_after` | MEASURED — **worse** |

### 1.1 The three facts most likely to be misread

1. **Grounding is reported under two protocols and two decode variants.** The trained head scores
   **0.2838** (canonical) and **0.2566** (matched6) mean best IoU; the head-argmax decode scores
   **0.1215** and the zero-shot baseline **0.0972**. Quoting one cell of that grid alone is selective.
   The **matched6** protocol is the decode-matched comparison (head emits 6 boxes, baseline emits
   5.99); the **canonical** protocol is the config default (top_k 20).
2. **Change-VQA is reported on two test sets.** `test` (0.697626 / 0.378373) and `test2`
   (0.651469 / 0.372309). Quoting only the better one would be selective.
3. **Router 0.965116 is validation, ungated, n = 86.** It is not a test result; the router test split
   was **NOT RUN** for the shipped artifact.

### 1.2 The only `VERIFIED` headline

The change-detection triple — **pooled IoU 0.8122 / macro IoU 0.8457 / pooled F1 0.8964** — is the
**only** headline in this repository that carries the `VERIFIED` tag, because it is the only one
measured against a single, immutable public test split with a frozen threshold. Everything else is
`MEASURED` (real, artifact-backed, but with an open ruling, a protocol caveat, or a validation-only
population).

---

## 2. Source artifacts

| Metric family | Artifact (paths relative to the source repository root) |
|---|---|
| change | `artifacts/change/eval_test/eval_result.json` |
| grounding (canonical) | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` |
| grounding (matched6) | `artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json` |
| optical-SAR | `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json` |
| change-VQA | `artifacts/change_vqa/run/PROMOTION.json` |
| router | `artifacts/router/threshold_sweep_val.json` |
| calibration | `artifacts/calibration_v001.json` |
| VLM | `artifacts/vlm/phase6_closure.json` |

Each of these artifacts carries the frozen config hash `78f1e3700da15aa1` (where it records a hash at
all), which ties it to the same `configs/base.yaml`. A config edit moves the hash and detaches the
number from its configuration — which is why the change head is wired into serving via a registry
override rather than a config key (`docs/MODELS.md` §3.1).

> **All 20 numeric claims in the headline table are checked against these files by
> [`../tools/verify_readme_metrics.py`](../tools/verify_readme_metrics.py). Its output
> (`ALL CLAIMS VERIFIED`) is committed as [`../tools/readme_metrics_report.txt`](../tools/readme_metrics_report.txt).**
> §8 documents exactly what that tool checks.

---

## 3. The honesty rules this document follows

These are not aspirations. Each one is a rule that changed what is reported, and each is checkable
against the artifacts.

1. **Evidence before claims.** Every reported number has an artifact path. A number with no artifact
   does not appear in this document.
2. **Two protocols are never collapsed.** Grounding is reported under *both* the canonical and
   matched6 protocols. Quoting 0.2838 alone would be selective; quoting 0.2566 alone would be too.
3. **Two decode variants are never collapsed into the trained-head claim.** `head_argmax` (0.1215) is
   **not** apples-to-apples with the multi-box baseline (mean best IoU is a max over predictions, so
   1 box vs ~6 boxes flatters the head). It is reported because it is the number comparable to the
   Phase 7 resolution experiment's zero-shot argmax — not because it decides anything.
4. **Two test sets are never collapsed.** Change-VQA is reported on `test` **and** `test2`.
5. **accuracy never travels without macro-F1.** For imbalanced multi-class heads (optical-SAR,
   change-VQA) the macro-F1 is reported alongside accuracy, always, in the same table, with the same
   split and the same `n`.
6. **Validation is not test.** The router figure is labelled "overall **ungated** accuracy", val,
   n = 86, corpus-limited. It is not a test result.
7. **A negative result stays negative.** Calibration ECE worsened (0.013755 → 0.014929) and is shown
   worsening. It is retained because it is in the frozen config, **not** because it helped.
8. **USABLE ≠ ACCEPTED.** The VLM adapter's metrics are real and reproducible; the artifact is
   nevertheless `ACCEPTANCE-REJECTED`. Both statements appear together.
9. **A metric is never promoted to a system-level claim.** There is no end-to-end benchmark, and none
   is invented by averaging the per-task numbers.
10. **An absent number is stated, not omitted.** §6 lists everything that is `NOT RUN`.
11. **A protocol-sensitive number carries its protocol.** Grounding always names canonical vs matched6
    and the decode variant; optical-SAR always names the single-label subset and the 19-slot
    denominator.
12. **A pre-registered rule is applied as written.** The grounding resolution decision was decided by a
    rule fixed *before* the result was seen, and the artifact records
    `rule_changed_since_preregistration: false` (`docs/MODELS.md` §4.1).
13. **A metric with a definitional detail states it.** The optical-SAR macro-F1 of 0.434161 is the mean
    over **all 19 class slots**, not the 14 present classes — averaging over the present classes gives
    0.589218, and the artifact records `macro_f1_denominator`, `classes_present`, `classes_absent` and
    the full 19-term vector so the difference cannot be misread.

---

## 4. Per-task evaluation protocols

### 4.1 Change detection — `VERIFIED`

- **Artifact:** `artifacts/change/eval_test/eval_result.json`.
- **Split:** LEVIR-CD-256 **test**, n = **2,048** (immutable public split).
- **Threshold:** **0.50**, frozen in `configs/base.yaml` (`change.threshold`).
- **Metrics:** pooled IoU / macro IoU / pooled F1, **plus the full confusion counts** so any metric can
  be recomputed rather than trusted.
- **Protocol detail:** tiles of 256 px (`change.tile_size: 256`), overlap 0
  (`change.tile_overlap: 0`), minimum component 32 px (`change.min_component_pixels: 32`).

**Why both pooled and macro.** The test split is only ≈ 5 % changed pixels
(`metrics.mean_change_fraction: 0.0509`), and the **median tile has zero changed pixels**
(`change_fraction_quantiles.p50: 0.0`). Pooled IoU (0.8122) and macro IoU (0.8457) answer different
questions about that imbalance.

**Measured numbers** (all key paths under `metrics`):

| Metric | Value | Key path |
|---|---|---|
| pooled IoU | **0.8122** | `pooled.iou` |
| pooled F1 | **0.8964** | `pooled.f1` |
| pooled mean IoU (`miou`) | 0.9007 | `pooled.miou` |
| pooled precision | 0.9195 | `pooled.precision` |
| pooled recall | 0.8745 | `pooled.recall` |
| macro IoU | **0.8457** | `macro.miou` |
| macro F1 | 0.7962 | `macro.f1` |
| macro IoU (`iou`, change-class IoU) | 0.718 | `macro.iou` |
| macro precision | 0.8506 | `macro.precision` |
| macro recall | 0.7757 | `macro.recall` |
| tp | 5,978,997 | `pooled.tp` |
| fp | 523,658 | `pooled.fp` |
| fn | 858,407 | `pooled.fn` |
| tn | 126,856,666 | `pooled.tn` |
| n_pixels | 134,217,728 | `pooled.n_pixels` |

**Population:** `n: 2048`, `n_images_with_change: 935`,
`change_fraction_quantiles: {min: 0.0, p50: 0.0, p90: 0.197205, max: 0.684937}`.

**Environment and integrity:** device `cuda`, torch `2.10.0+cu128`, Python `3.12.13`,
`seconds: 55.359`; `config_hash: 78f1e3700da15aa1`, `checkpoint_config_hash: 78f1e3700da15aa1`,
`checkpoint_config_hash_checked: true`, `config_drift: false`.

**Status:** `VERIFIED`. See [`EVALUATION.md`](EVALUATION.md) §2.1 and [`MODELS.md`](MODELS.md) §3.1.

---

### 4.2 Grounding — `MEASURED`, two protocols × two decode variants

- **Artifacts:** `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`
  (`head_decode.top_k: 20`) and `…/eval_result_matched6.json` (`head_decode.top_k: 6`).
- **Split:** VRSBench **eval** split, n = **16,159** records, scored once, no sampling
  (`n_eval_records: 16159`, `limited_run: false`).
- **Resolution:** **224** (`resolution: 224`, `grid: 7`), frozen; 448 rejected by a pre-registered
  paired test (`docs/MODELS.md` §4.1).
- **Box convention:** VRSBench stores boxes normalised to **0–100**; this project stores **0–1**, via
  the declared `grounding.benchmark_box_scale: 100.0` so the conversion cannot be applied twice.
- **Protocols:** *canonical* and *matched6* — both reported.
- **Decode variants:** *head threshold* (the shipped decode), *head argmax*, and *zero-shot matched*
  (the baseline).

**The `head_decode` block is recorded in each artifact** — `config_default_top_k: 20`,
`nms_iou: 0.5`, `score_threshold: 0.4`, `top_k: 20` (canonical) / `top_k: 6` (matched6). Without it
the artifact could not say which setting produced its number, and a re-run at the config default would
silently yield a different figure.

**The full grid — canonical** (`eval_result_canonical.json` → `results`):

| Strategy | `mean_best_iou` | `recall.0.10` | `recall.0.25` | `recall.0.50` | `latency_ms_per_image` | `seconds` |
|---|---|---|---|---|---|---|
| `head_argmax` | **0.1215** | 0.3183 | 0.2088 | 0.0795 | 0.655 | 10.6 |
| `head_threshold` (top_k 20) | **0.2838** | 0.6882 | 0.5047 | **0.2198** | 2.205 | 35.6 |
| `zero_shot_matched` | **0.0972** | 0.3298 | 0.1188 | 0.0234 | — | 17.9 |

**The full grid — matched6** (`eval_result_matched6.json` → `results`):

| Strategy | `mean_best_iou` | `recall.0.10` | `recall.0.25` | `recall.0.50` | `latency_ms_per_image` | `seconds` |
|---|---|---|---|---|---|---|
| `head_argmax` | **0.1215** | 0.3183 | 0.2088 | 0.0795 | 0.652 | 10.5 |
| `head_threshold` (top_k 6) | **0.2566** | 0.6315 | 0.4545 | **0.1938** | 2.158 | 34.9 |
| `zero_shot_matched` | **0.0972** | 0.3298 | 0.1188 | 0.0234 | — | 15.5 |

Both artifacts carry `phase7_reference: {mean_best_iou: 0.0972, recall_at_0.50: 0.0234,
source: "docs/PHASE7_RESOLUTION_DECISION.md"}` — the recorded baseline against which the head is
judged.

**Decode-matched delta (head threshold top_k = 6 vs zero-shot): +0.1594 IoU, +0.1704 Recall@0.50.**
The pre-registered bar was `MIN_IMPROVEMENT_IOU = 0.02`; the measured margin is **8× the bar**.

**Why `head_argmax` is not the trained-head claim.** 1 box against ~6 boxes flatters the head, because
mean best IoU takes the max over predictions. It is reported for comparability with the Phase 7
resolution experiment, not as the head's result.

**A decode defect this protocol had to fix first** (`docs/PHASE8_GROUNDING_HEAD_DECISION.md`). The
evaluation script originally built its own single-box baseline with `argmax_candidate`, while Phase 7
measured the baseline through `ground_phrase` (threshold box + up to 5 local maxima). Same 16,159
records, same metric, same cached features — **different decode**: 0.0972 vs 0.0092 (a 10× gap). The
eval printed `head beats zero-shot: True (+0.1123)` when the matched comparison was +0.0243. The fix
extracted the decode into one function both paths call
(`specialists/grounding/inference.py::decode_candidates_from_features`), and re-running the eval
reproduced the Phase 7 number **exactly** (`|difference| = 0.0000`) — proof the fix is real, not
cosmetic.

**Status:** `MEASURED`. The trained head beats the zero-shot baseline; the absolute IoU is low, which
is an honest limitation. See [`EVALUATION.md`](EVALUATION.md) §2.2 and [`MODELS.md`](MODELS.md) §3.4.

---

### 4.3 Optical-SAR fusion — `MEASURED`, ruling `OPEN`

- **Artifact:** `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`.
- **Tool:** `scripts/eval_fusion_115.py` (read-only) — a **separate, later** step from training.
- **Split:** held-out **test**, n = **4,000** (`n_scored: 4000`, `split: "test"`).
- **Label space:** 19 CLC classes; **14 present, 5 absent** in the scored split.
- **macro-F1 denominator:** all 19 classes (absent classes contribute 0.0) — recorded explicitly.

**Why the metric is computed by a separate tool.** Every `run_record.json` carries
`pre_registered_metric_computed = false` and `result_status = "PLUMBING_ONLY — fixture/loop evidence,
NOT a result; the pre-registered 11.5 metric is not computed"`. That flag is **correct about the
trainer** and must not be "fixed": the trainer fits on `train`/`val` and never opens the test split,
deliberately, so the held-out split cannot be contaminated by the search over 10 runs, 2 arms and 5
seeds. The 11.5 metric therefore belongs to a separate, read-only step.

**The tool refuses (exit 2) rather than guessing** when: the head or cache is missing; the split is
empty; the cache `config_hash` is not `78f1e3700da15aa1` (a different experiment); or the head cannot
be built against the frozen `(B, 2318)` input contract.

**Measured numbers:**

| Metric | Value | Exact key path |
|---|---|---|
| **accuracy** | **0.931** | `accuracy` |
| **macro F1** | **0.434161** | `macro_f1` |
| loss | 0.254592 | `loss` |
| n scored | 4,000 | `n_scored` |
| num classes | 19 | `num_classes` |
| macro-F1 denominator | all 19 classes (absent contribute 0.0) | `macro_f1_denominator` |
| classes present | `[0,2,3,4,5,6,7,8,9,10,12,13,17,18]` | `classes_present` |
| classes absent | `[1,11,14,15,16]` | `classes_absent` |
| deciding statistic | **false** | `is_deciding_statistic` |
| head sha256 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` | `head_sha256` |
| head bytes | 14,427,457 | `head_bytes` |
| cache arm | A | `cache_arm` |
| cache config hash | `78f1e3700da15aa1` | `cache_config_hash` |

**Optical-SAR accuracy 0.931 must ALWAYS travel with macro-F1 0.434161.** The two are recorded side by
side in the artifact precisely so neither can be quoted alone.

**Per-class F1 (`_per_class_f1`, 19 terms):**

```
[1.000, 0.000, 0.428, 0.571, 0.887, 0.000, 0.000, 0.800, 0.929, 0.940,
 0.182, 0.000, 0.200, 0.438, 0.000, 0.000, 0.000, 0.877, 0.998]
```

**Two averages from the same vector — the denominator trap:**

| Averaged over | Value |
|---|---|
| **all 19 slots** — the pre-registered definition | **0.434161** |
| the 14 present classes only | **0.589218** |

A reader who computes the second and compares it to the recorded scalar will wrongly conclude the
recorded figure is wrong. It is the 19-slot mean by construction. Reproducible:
`sum(per_class)/19 = 0.43416082670034284`, `sum(per_class[c] for c in present)/14 = 0.5892182648076082`
(`docs/PHASE12_115_METRIC_COMPUTED.md` §3.5).

**Five absent classes contribute `0.0` by definition — but two PRESENT classes also score `0.0`.**
Classes **1, 11, 14, 15, 16** are absent; classes **5** and **6** are **present** and score `0.0` — a
**real total miss**. So part of the low macro-F1 is populational, and part is genuine per-class failure.
**High accuracy with a wide per-class spread is the signature of prediction dominated by frequent
classes.**

**Measured test-set label distribution** (`docs/PHASE12_115_METRIC_COMPUTED.md` §3.4):

```
[13, 0, 63, 39, 846, 4, 1, 6, 453, 246, 4, 0, 9, 18, 0, 0, 0, 34, 2264]
```

14 of 19 classes present; five have zero samples; top-to-bottom ratio **2,264 : 1** (class 18 vs class
6). The majority class holds 2,264 / 4,000 = **0.566**; the head scores **0.931**, so the accuracy is
not a constant predictor. Per-class recall (selected): class 18 n=2264 → **0.997**; class 9 n=246 →
0.951; class 17 n=34 → 0.941; class 8 n=453 → 0.905; class 4 n=846 → 0.809; class 3 n=39 → 0.410;
class 12 n=9 → 0.111; classes 5 and 6 → **0.000**.

**Four independent verification checks** (`docs/PHASE12_115_METRIC_COMPUTED.md` §3): (1)
`evaluate_fusion_head` directly returns `n=4000, accuracy=0.931, macro_f1=0.43416082670034284`; (2)
rebuilding the input tensor by hand — `concat([optical_gap, sar_gap, joint_gap])` then
`concat([…, optical_mask, sar_mask])`, widths `(4000,768)×3 + (4000,12) + (4000,2) = 2318` — gives
0.931 again (this checks feature **ordering**); (3) `sklearn.metrics` `accuracy_score = 0.931`,
`f1_score(average="macro", labels=range(19), zero_division=0) = 0.43416082670034284`; (4) the per-class
breakdown above.

**⚠️ The caveat that governs how these numbers may be used.** The cache metadata records
`label_policy = require_single_label`, `n_skipped_by_policy = 0`. reBEN v2.0 is a **multi-label**
corpus while the frozen head is a **single-label 19-class softmax**. The extraction used policy (b) —
restrict to single-label patches — which preserves the frozen architecture exactly but **changes the
evaluation population**. Single-label patches are **17.57 %** of the corpus (96,537 of 549,488), and the
rarest class survives as **1 patch** (a 59,204 : 1 imbalance). So this metric **may not** be presented
as a multi-label BigEarthNet/reBEN result, as comparable to published BigEarthNet numbers, or as a
statement about all 19 classes. It **may** be presented as the pre-registered 11.5 metric under the
single-label extraction policy on the held-out split.

**Status:** `MEASURED`, ruling **OPEN**. `is_deciding_statistic: false`; the artifact's `advisory`
states it *"selects no head, ranks nothing and compares no arms. Whether this constitutes a Phase 12
pass is the owner's ruling."* No numerical threshold is attached. See [`EVALUATION.md`](EVALUATION.md)
§2.3 and [`MODELS.md`](MODELS.md) §3.3.

---

### 4.4 Change-VQA — `MEASURED`, ruling `OPEN`

- **Artifact:** `artifacts/change_vqa/run/PROMOTION.json` (`verification` block).
- **Splits:** `test` (n = **39,686**) and `test2` (n = **31,036**).
- **Selection:** epoch **8**, chosen on **val answer accuracy 0.700018**, stopped by early stopping
  (`identity.selected_on`, `identity.stop_reason`).
- **Answer space:** a **closed 19-answer space** over the eight CDVQA question types; type mask applied
  by default (`verification.mask_gain: 0.0`).

**Two test sets are reported. Quoting only the better one would be selective.**

| Test set | n | accuracy | macro F1 |
|---|---|---|---|
| `test` | 39,686 | **0.697626** | **0.378373** |
| `test2` | 31,036 | **0.651469** | **0.372309** |

| Field | Value | Exact key path |
|---|---|---|
| test accuracy | 0.697626367 | `verification.test_accuracy` |
| test macro F1 | 0.378373275 | `verification.test_macro_f1` |
| test2 accuracy | 0.651469262 | `verification.test2_accuracy` |
| test2 macro F1 | 0.372308516 | `verification.test2_macro_f1` |
| n scored, test | 39,686 | `verification.n_scored_test` |
| n scored, test2 | 31,036 | `verification.n_scored_test2` |
| global majority baseline, test | 0.311546 | `verification.global_majority_baseline_test` |
| global majority baseline, test2 | 0.178728 | `verification.global_majority_baseline_test2` |
| mask gain | 0.0 | `verification.mask_gain` |
| checks passed / failed / unverified | 93 / 0 / 0 | `verification.checks_*` |
| metric ruling | `OPEN — the plan leaves the accuracy/macro-F1 interpretation owner-gated. No official aggregate metric is asserted here.` | `verification.metric_ruling` |

**The gap between accuracy and macro-F1 means the head is carried by common classes.** The
global-majority baselines make it concrete: 0.311546 on `test` means a constant predictor scores 0.31
and the head scores 0.70 — but the macro-F1 of 0.378 shows the per-class picture is far weaker than the
aggregate.

**Status:** `MEASURED`, ruling **OPEN**. No promotion/acceptance decision has been recorded
(`PROMOTION.json` → `state.note`). See [`EVALUATION.md`](EVALUATION.md) §2.4 and
[`MODELS.md`](MODELS.md) §3.2.

---

### 4.5 VLM adapter — `MEASURED`, `ACCEPTANCE-REJECTED`

- **Artifact:** `artifacts/vlm/phase6_closure.json` (`status: CLOSED`).
- **Split:** a frozen **1,000-question subset** of the test split, whose identity was proven without a
  model (Gate A″): `available_per_split {val: 6750, test: 7772}`, subset `n = 1000`, test per-class
  counts an **exact** match to the recovery subset (19 classes, sum 1000).

**The "usable" numbers** (`why_usable_verified.adapted_test`):

| Metric | Value | Exact key path |
|---|---|---|
| **exact_match** | **0.963** | `why_usable_verified.adapted_test.exact_match` |
| **F1** | **0.96432** | `why_usable_verified.adapted_test.f1` |
| n | 1000 | `…adapted_test.n` |
| precision | 0.963391 | `…adapted_test.precision` |
| recall | 0.965251 | `…adapted_test.recall` |
| tp / fp / tn / fn | 500 / 19 / 463 / 18 | `…adapted_test.confusion` |
| aggregate test delta | **+49.50 pp** (46.80 → 96.30 pp) | `why_usable_verified.aggregate_test_delta_pp` |

**The "rejected" side** (`why_acceptance_rejected`, rule version **v002**, decision split **test**):

| | |
|---|---|
| **V1** | **passes** — test `46.80 → 96.30 pp`, delta **+49.50 pp** (required ≥ +5.00) |
| **V2** | **fails** — **Mixed forest**: `n=33`, `100.00 → 87.8788 pp`, `drop 12.1212 pp`, `lost_questions 4`, `z 2.1335` |

It fails **both** halves of v002 (`lost ≥ 4` **and** `z ≥ 1.96`). Per item V, a complete run that fails
V2 is `REJECTED`. The rejection is **narrow**: `n_classes_failed: 1`, `n_classes_held: 5`,
`n_classes_improved: 11`, `n_classes_total: 19`. The next-worst class (Inland wetlands, −6.6667 pp,
n=30) lost only 2 questions and sits below V2's materiality floor.

**It is not a split artefact.** The same class also degraded on **val** in Run 1 (drop 6.4516 pp,
n=31). The adapter hurts Mixed forest on **both** splits. Mixed forest also sits at a 100.00 pp baseline
on test, so any loss is a drop from the ceiling.

**Residual risk, reported not resolved.** The verdict rests on **4 questions in one class of 33**. With
no `n ≥ N` floor in V2, a 33-question class can flip the verdict of a run whose aggregate endpoint
improved by 49.5 pp.

**Both rejection records are preserved** (`preserved_records`): Run 1's manifest `v001` on **val**
`REJECTED` (3 classes, `val_delta_pp 42.0`, `test_delta_pp null`); the independent test-split
adjudication `v002` on **test** `REJECTED`; and the recovery manifest `v002` on **val** `ACCEPTED` —
recorded **for completeness only, NOT final acceptance**, because it decides on the same val subset
that motivated v002.

**Status:** `MEASURED`; artifact `CLOSED`; headline **`ACCEPTANCE-REJECTED`**. **USABLE ≠ ACCEPTED.**
The deployed caption/VQA path uses the unadapted model. See [`EVALUATION.md`](EVALUATION.md) §2.5 and
[`MODELS.md`](MODELS.md) §3.6.

---

### 4.6 Router — `MEASURED`, `TEST NOT RUN`

- **Artifact:** `artifacts/router/threshold_sweep_val.json`.
- **Split scored:** **validation**, n = **86** (`n_val: 86`, `n_val_examples_scored: 86`,
  `n_test_examples_scored: 0`, `test_split_touched: false`).
- **Metric:** overall **ungated** accuracy **0.965116** (`overall_ungated_accuracy`).
- **Corpus:** **corpus-limited** (`corpus_limited: true`); `corpus_total: 576`, `corpus_groups: 54`;
  plan minima `plan_min_val_queries: 500`, `plan_min_hard_negatives: 100`; `val_min_support: 8`;
  `hard_negatives_in_val: 0` (by design — `hn_*` families are held out to TEST).
- **Sweep:** thresholds 0.50 → 0.99; `select_by: "covered_accuracy"`;
  `selected.threshold: 0.76` (coverage 0.790698, covered-task accuracy 1.0);
  `shipped_threshold: 0.70` (coverage 0.848837, covered-task accuracy 0.972603);
  `delta_vs_shipped: {coverage: -0.0581, covered_task_accuracy: 0.0274}`.
- **Split sizes:** `{train: 410, test: 80, val: 86}`.

**The artifact's own note, quoted verbatim in substance:** *"corpus-limited: val n=86 vs plan >=500.
This is NOT a calibration — the corpus is synthetic and too small (min per-class support 8, caption)
and val carries 0 hard negatives (hn_* families are held out to TEST by design). Selecting a threshold
here yields a justified default, not a calibrated value. The corpus was NOT padded with generated
queries. Backlog P1-9's 'n=80' is the TEST split; the sweep target is val n=86. **The test split was
NOT touched.**"*

**The router TEST split was NOT RUN.** Do not read 0.965116 as a test result. The sweep scored 86
validation examples and **zero** test examples.

> **Historical record, kept distinct.** `docs/PHASE4_ROUTER_REPORT.md` (verified 2026-09-16, Gate 2
> PASS) records an **earlier** gate-2 evaluation that included a test column — test task accuracy
> **0.975**, macro F1 0.976, hard-negative accuracy **0.800**. That evaluation predates the shipped
> threshold sweep, and its own standing caveat is that *"the 0.975 headline is partly earned on
> templates the split kept in training. Treat the router as working, not as benchmarked."* The release's
> position for the shipped artifact is **TEST NOT RUN**; the Phase 4 numbers are retained as a
> historical record, not promoted to a release benchmark.

**Status:** `MEASURED` — **TEST NOT RUN**. See [`EVALUATION.md`](EVALUATION.md) §2.6 and
[`MODELS.md`](MODELS.md) §3.5.

---

### 4.7 Calibration — `MEASURED`, negative result

- **Artifact:** `artifacts/calibration_v001.json` (`schema: calibration_v1`).
- **Fit split:** **val**, n = **16,441** (`temperature_scaling.fitted_on: "Val"`,
  `temperature_scaling.n_samples: 16441`).
- **Temperature:** **T = 0.9772731820958189** (`temperature_scaling.temperature`).
- **Scope:** the R-02 change-VQA head's answer confidence only; other specialists emit their own raw
  scores and are unaffected (`scope.note`).

| Metric | Before | After | Improvement | Key path |
|---|---|---|---|---|
| **ECE** | **0.013755** | **0.014929** | **−0.001174** | `metrics.ece_before` / `metrics.ece_after` / `metrics.ece_improvement` |
| NLL | 0.689741 | 0.689631 | +0.00011 | `metrics.nll_before` / `metrics.nll_after` |

`ece_improvement = −0.001174` is **negative**: expected calibration error **increased**. NLL improved
by a negligible 0.00011 (`fit_diagnostics.nll_improvement: 0.0001103574982127542`).
`metrics.n_bins: 15`, `metrics.n_classes: 19`.

**The scaling is retained because it is part of the frozen configuration, not because it helped.**
`provenance.held_out_splits_excluded: ["Test", "Test2"]` — the temperature was fitted only on val.

**The reliability diagram is explicitly labelled pre-scaling.** `reliability_diagram.ece` is
**0.013755** (the *before* value), with the note: *"Equal-width bins over predicted-class confidence.
ECE is bin-count sensitive and is not an aggregate score."* The diagram on the Benchmark page is
labelled **pre-scaling** so a reader cannot mistake it for the calibrated result.

**Status:** `MEASURED` — **worse**. A negative result, kept negative. See [`EVALUATION.md`](EVALUATION.md)
§2.7 and [`MODELS.md`](MODELS.md) §5.

---

## 5. Live validation — behavioural, not accuracy

Accuracy is separate from **behavioural** validation. The deployed stack was driven end-to-end in a
headed browser, one upload per case, with per-case screenshots and recorded run ids.

| Property | Result |
|---|---|
| Independent full passes | **3** |
| Cases per pass | 8 (6 regression + 2 router-defect) |
| Passes at 8/8 | **3 of 3** |
| Live runs executed | **24** |
| Correct dispatches | **24** |
| Mock-node contamination | **0** on every live run |
| Trace fill | **94.4444 %** on every live run |
| Frontend regression suite | **106 passed** (`tests/unit/test_frontend_live_wiring.py`) |

**3 passes × 8 cases = 24 runs, 8/8 each, 0 mock nodes, 94.4444 % trace fill.** Every pass produced
**fresh run identifiers** — no run id is shared between passes. This is behavioural evidence that the
pipeline runs and routes correctly; it is **not** an accuracy claim.

### 5.1 Harness integrity — a false-pass bug that was caught

An earlier harness revision typed queries with **synthetic CDP key events**, which Chrome **silently
drops when the window lacks OS focus**. The harness therefore dispatched the page's *default* query and
still recorded a "result" — a **false pass**.

The current harness **asserts form state before dispatch**: that the query box really holds the intended
query, that `#obsTail` reads `ready`, and that both frames are attached for pair tasks. The earlier 8/8
run was independently checked and confirmed **not** to have been infected (its answers were
query-specific and the query text was embedded in the answers). This failure mode is recorded because
it is exactly the kind of silent false-positive an evaluation harness must not have.

### 5.2 Deployed HEADs

The live validation ran against these deployed revisions: Frontend `2d7ae53b482d`,
Backend `89d80eaddec5`, Inference `5a0936ace491`.

### 5.3 What live validation does and does not establish

- **Establishes:** the pipeline runs end-to-end; the router dispatches the right specialist for the 8
  cases in each pass; no mock nodes contaminate a live run; the trace is populated to 94.4444 %.
- **Does not establish:** any accuracy figure. There is no labelled ground truth attached to the 24
  live runs, so no accuracy can be computed from them.

---

## 6. What is NOT benchmarked

An exhaustive list. Absent numbers are stated, never silently omitted or replaced with an estimate.

| Benchmark | Status | Note |
|---|---|---|
| **System-level end-to-end accuracy** | **NOT RUN — none exists** | There is no measured end-to-end benchmark of the full router → specialist → envelope pipeline. No such number is claimed anywhere. |
| **Router test split** | **NOT RUN** | Only the validation split (n = 86) was scored by the shipped threshold sweep. |
| **Grounding at 448 for the trained head** | **NOT RUN** | Only the *zero-shot baseline* was measured at 448 (`docs/MODELS.md` §4.1). Re-opening the resolution question for the trained head would be a new pre-registered experiment. |
| **Benchmark adapters** | **NOT RUN** | Adapter-based benchmark runs were not executed. |
| **Efficiency / latency benchmark** | **NOT RUN (system-level)** | Per-specialist latency is recorded incidentally in artifacts (e.g. grounding `latency_ms_per_image` 2.205 ms for the threshold decode, 0.655 ms for argmax; CROMA forward pass 0.89 s for a batch of 2 at 120 px), but there is **no end-to-end latency benchmark**. |
| **Cross-dataset generalisation** | **NOT RUN** | Each specialist is evaluated only on its own training-family test split. |
| **Human evaluation** | **NOT RUN** | No human rating of answers was performed. |
| **Robustness / adversarial evaluation** | **NOT RUN** | No perturbation, adversarial, or out-of-distribution robustness study was performed. |
| **Multi-label BigEarthNet / reBEN result** | **NOT PRODUCED** | The local subset is 100 % single-label against the official 1–11 multi-label scheme. Metrics are **not comparable** to published multi-label numbers. |
| **Grounding head wired and benchmarked in the serving path** | **NOT RUN** | `specialists/grounding/inference.py` exposes only the zero-shot path as of the Phase 8 record; wiring the head is the remaining integration step. |
| **Change-VQA served end-to-end** | **NOT RUN** | Serving needs both a trained head and a trained feature extractor; neither is wired by default, and an untrained head refuses to answer. |
| **Optical-SAR multi-label evaluation** | **NOT RUN** | See the single-label caveat (§4.3). |
| **Change-VQA on a third held-out set** | **NOT RUN** | Only `test` and `test2` exist. |
| **Any private / hidden evaluation data** | **NOT ACCESSED** | `evaluation.hidden_data_access: false` in `configs/base.yaml`. |
| **Calibrated reliability curve** | **NOT PLOTTED** | Only the pre-scaling diagram exists; the calibrated curve is not plotted. |
| **Per-class support for the router test split** | **NOT RUN** | The test split was not scored, so it has no support table. |

---

## 7. How to reproduce

### 7.1 Verify every number in this document against its artifact

```bash
# verify every README/benchmark number against its source artifact
python release/tools/verify_readme_metrics.py
```

The script resolves nested artifact keys — including keys that themselves contain dots, such as the
`recall` dict keyed `"0.10"/"0.25"/"0.50"` — and compares each value at the precision printed in the
README. It exits **non-zero if any claim fails**, and prints `ALL CLAIMS VERIFIED` (exit 0) only when
all 20 numeric claims match and the status assertions hold. Committed output:
[`../tools/readme_metrics_report.txt`](../tools/readme_metrics_report.txt).

### 7.2 Reproduce the individual metrics

| Metric | Reproduction path |
|---|---|
| change | `scripts/eval_change.py` (refuses to score on a config-hash drift, exit 3) |
| grounding | `scripts/exp_grounding_resolution.py` + `scripts/analyze_grounding_resolution.py`; the two eval JSONs are recomputable from the per-sample JSONL without re-running the encoder |
| optical-SAR | `python scripts/eval_fusion_115.py --head … --cache-dir artifacts/optical_sar/fusion_features --split test --report …` (exit 0 = computed; exit 2 = refused) |
| change-VQA | `scripts/evaluate_change_vqa.py` (default checkpoint `artifacts/change_vqa/run/head.pt`) |
| VLM | `scripts/phase6_close.py` regenerates the closure record from the evidence; `scripts/phase6_adjudicate_test.py` re-runs the independent test-split adjudication |
| calibration | `load_calibration(config, base_dir='configs')` consumes `calibration_v001.json` |
| router | `scripts/train_router.py` (cached-embedding training); the sweep artifact is `threshold_sweep_val.json` |

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full reproduction guide and
[`EVALUATION.md`](EVALUATION.md) for how each number was produced.

### 7.3 Reproduce the live validation

The 24 live runs were driven in a headed browser with per-case screenshots and recorded run ids. The
harness asserts form state before dispatch (§5.1). See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for
the harness entry point.

### 7.4 Verify the released artifacts

```bash
sha256sum -c models/checksums.sha256
```

---

## 8. The verification tool — exactly what it checks

`release/tools/verify_readme_metrics.py` is read-only: it writes nothing and prints a report suitable
for pasting into a release-verification record. It holds a hard-coded `CLAIMS` list of
`(label, artifact path, dotted key, printed value)` tuples. Key names were **discovered by walking the
artifacts, not assumed** — several live under nested paths (change metrics are `metrics.pooled.iou`;
grounding is `results.head_threshold.mean_best_iou`; VLM is `why_usable_verified.adapted_test.*`).

### 8.1 The 20 numeric claims

| # | Claim | Artifact | Key path | Printed |
|---|---|---|---|---|
| 1 | change pooled IoU | `artifacts/change/eval_test/eval_result.json` | `metrics.pooled.iou` | 0.8122 |
| 2 | change macro IoU | `…/eval_result.json` | `metrics.macro.miou` | 0.8457 |
| 3 | change pooled F1 | `…/eval_result.json` | `metrics.pooled.f1` | 0.8964 |
| 4 | grounding canonical head_threshold mean_best_IoU | `…/eval_result_canonical.json` | `results.head_threshold.mean_best_iou` | 0.2838 |
| 5 | grounding canonical head_threshold recall@0.5 | `…/eval_result_canonical.json` | `results.head_threshold.recall.0.50` | 0.2198 |
| 6 | grounding matched6 head_threshold mean_best_IoU | `…/eval_result_matched6.json` | `results.head_threshold.mean_best_iou` | 0.2566 |
| 7 | grounding matched6 head_threshold recall@0.5 | `…/eval_result_matched6.json` | `results.head_threshold.recall.0.50` | 0.1938 |
| 8 | grounding head_argmax mean_best_IoU (canonical) | `…/eval_result_canonical.json` | `results.head_argmax.mean_best_iou` | 0.1215 |
| 9 | grounding zero-shot baseline IoU (canonical) | `…/eval_result_canonical.json` | `results.zero_shot_matched.mean_best_iou` | 0.0972 |
| 10 | optical-SAR fusion accuracy | `…/pre_registered_115_metric.json` | `accuracy` | 0.931 |
| 11 | optical-SAR fusion macro_F1 | `…/pre_registered_115_metric.json` | `macro_f1` | 0.434161 |
| 12 | change_vqa test accuracy | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test_accuracy` | 0.697626 |
| 13 | change_vqa test macro_F1 | `…/PROMOTION.json` | `verification.test_macro_f1` | 0.378373 |
| 14 | change_vqa test2 accuracy | `…/PROMOTION.json` | `verification.test2_accuracy` | 0.651469 |
| 15 | change_vqa test2 macro_F1 | `…/PROMOTION.json` | `verification.test2_macro_f1` | 0.372309 |
| 16 | router overall ungated accuracy | `artifacts/router/threshold_sweep_val.json` | `overall_ungated_accuracy` | 0.965116 |
| 17 | calibration ECE before scaling | `artifacts/calibration_v001.json` | `metrics.ece_before` | 0.013755 |
| 18 | calibration ECE after scaling | `artifacts/calibration_v001.json` | `metrics.ece_after` | 0.014929 |
| 19 | VLM adapter exact_match | `artifacts/vlm/phase6_closure.json` | `why_usable_verified.adapted_test.exact_match` | 0.963 |
| 20 | VLM adapter F1 | `artifacts/vlm/phase6_closure.json` | `why_usable_verified.adapted_test.f1` | 0.96432 |

### 8.2 The status assertions

Beyond the 20 numeric claims, the tool asserts artifact-declared **statuses**:

```
VLM headline contains ACCEPTANCE-REJECTED : True
VLM status                               : CLOSED
router corpus_limited                    : True
router n_val                             : 86
calibration temperature                  : 0.9772731820958189
calibration ece_improvement              : -0.001174  (negative => calibration did NOT help)
```

### 8.3 The committed output

`release/tools/readme_metrics_report.txt` records all 20 rows as `MATCH`, all six status assertions as
expected, and the final line:

```
RESULT: ALL CLAIMS VERIFIED
```

The tool exits `0` on that result and `1` otherwise, so a drifted artifact fails the release check
rather than passing silently.

### 8.4 What the tool does not check

- It does **not** check the `n` or `split` fields. A value could match while the population changed;
  the `n`/`split` columns in §1 are therefore maintained by hand against the artifacts.
- It does **not** check the calibration reliability diagram, the grounding per-sample JSONL, or the
  VLM per-class tables.
- It does **not** verify the live-validation record (§5), which is behavioural and not artifact-scored.

---

## 9. Artifact schemas and key-path resolution

Every metric in §1 is addressed by a dotted **key path** into a JSON artifact. Two mechanics matter,
because getting either wrong produces a `NOKEY` failure that looks like a missing metric rather than a
bug in the reader.

### 9.1 The dotted-key problem

Some artifact keys themselves contain dots. The grounding artifacts key their recall dict as
`{"0.10": …, "0.25": …, "0.50": …}`. A naive `dotted.split(".")` walk would break
`results.head_threshold.recall.0.50` into `… recall → 0 → 50` and fail — even though the value exists.

`verify_readme_metrics.py` resolves this by **preferring the longest matching key at each step**: at a
dict node it tries the longest possible joined key first, and only shortens when no key matches. That
is why the recall claim is written `results.head_threshold.recall.0.50` and resolves to the `"0.50"`
entry rather than to a nested `0` object.

### 9.2 Top-level schema of each artifact family

These are the top-level keys actually present in each artifact, read from the files:

| Artifact | Top-level keys (abridged to the load-bearing ones) |
|---|---|
| `change/eval_test/eval_result.json` | `artifact`, `checkpoint`, `checkpoint_config_hash`, `checkpoint_config_hash_checked`, `checkpoint_embedded_config`, `config_drift`, `config_hash`, `created_at`, `device`, `environment`, `metrics`, `n`, `n_images_with_change`, `change_fraction_quantiles`, `per_image_change_fraction` |
| `grounding/…/eval_result_canonical.json` | `checkpoint_config_hash`, `config_drift`, `config_hash`, `created_at`, `device`, `environment`, `eval_cache`, `frozen_config_evaluation`, `grid`, `head_decode`, `limited_run`, `n_eval_records`, `phase7_reference`, `resolution`, `results` |
| `grounding/…/eval_result_matched6.json` | same shape; `head_decode.top_k` differs (6 vs 20) |
| `optical_sar/…/pre_registered_115_metric.json` | `tool`, `metric`, `definition`, `split`, `head_path`, `head_sha256`, `head_bytes`, `cache_path`, `cache_arm`, `cache_config_hash`, `n_scored`, `accuracy`, `macro_f1`, `loss`, `num_classes`, `macro_f1_denominator`, `classes_present`, `classes_absent`, `_per_class_f1`, `is_deciding_statistic`, `advisory` |
| `change_vqa/run/PROMOTION.json` | `schema`, `promoted_utc`, `promoted_by`, `artifact`, `source`, `identity`, `frozen_dependency`, `verification`, `state`, `serving_wiring` |
| `router/threshold_sweep_val.json` | `adapter_config_hash`, `adapter_encoder`, `adapter_path`, `artifact`, `config_hash`, `corpus_groups`, `corpus_limited`, `corpus_total`, `created_at`, `delta_vs_shipped`, `device`, `encoder_type`, `environment`, `force`, `hard_negatives_in_val`, `n_test_examples_scored`, `n_val`, `n_val_examples_scored`, `note`, `overall_ungated_accuracy`, `plan_min_hard_negatives`, `plan_min_val_queries`, `rows`, `seconds`, `select_by`, `selected`, `shipped_row`, `shipped_threshold`, `split`, `split_sizes`, `test_split_touched`, `thresholds`, `val_min_support`, `val_task_counts` |
| `calibration_v001.json` | `consumer_contract`, `created_utc`, `fit_diagnostics`, `metrics`, `provenance`, `reliability_diagram`, `schema`, `scope`, `temperature_scaling`, `type_mask_applied` |
| `vlm/phase6_closure.json` | `artifact_verification`, `evidence_index`, `forward_rule`, `generated_at`, `headline`, `kind`, `not_a_retrain`, `phase`, `preserved_records`, `production_adapter`, `rule_unchanged_by_closure`, `status`, `verified_vs_accepted`, `what_closure_does_not_claim`, `why_acceptance_rejected`, `why_usable_verified` |

### 9.3 Status and provenance fields carried by every artifact

The artifacts are not just numbers; each carries the fields that make its number auditable:

| Field | Meaning | Where |
|---|---|---|
| `config_hash` | the frozen registry hash the run was made against (`78f1e3700da15aa1`) | change, grounding, optical-SAR, change-VQA, router, calibration, VLM |
| `checkpoint_config_hash` | the config hash **baked into the weights** | change, grounding |
| `config_drift` | whether the two hashes disagreed at evaluation time | change, grounding (both `false`) |
| `created_at` / `generated_at` | when the artifact was produced | all |
| `device` / `environment` | device and library versions | change, grounding, router |
| `split` | the population scored | optical-SAR (`test`), router (`val`) |
| `n` / `n_scored` / `n_eval_records` | the population size | all |

**Why `config_drift: false` matters.** The evaluation is only valid against the configuration the
weights were trained under. `checkpoint_config_hash_checked: true` and `config_drift: false` together
record that the check was *performed* and *passed* — not merely omitted.

### 9.4 The `is_deciding_statistic` and ruling fields

Two artifacts carry an explicit statement that their number is **not** a decision:

- optical-SAR: `is_deciding_statistic: false`, with an `advisory` that the tool *"selects no head,
  ranks nothing and compares no arms."*
- change-VQA: `verification.metric_ruling: "OPEN — … No official aggregate metric is asserted here."`

These fields exist so that a later reader cannot promote a measurement into a ruling by quoting it out
of context. Where a ruling exists, it is recorded separately (e.g. the optical-SAR A/B arm ruling was
made on `best_val_accuracy` — A 0.837100 vs B 0.839100, floor 0.0285 → **Arm A retained**, ruling
R-14 — not on the 0.931/0.434161 held-out metric).

---

## 10. Metric glossary and aggregation arithmetic

Definitions, so that a reader can recompute every number from the stored confusion counts or per-class
vectors.

### 10.1 Pooled vs macro

| Term | Definition | Where used |
|---|---|---|
| **pooled IoU** | IoU over the union of all pixels, computed from the global `tp/fp/fn` counts | change (`metrics.pooled.iou` = 0.8122) |
| **macro IoU** | unweighted mean of per-class IoU. For binary change this is the mean over {change, no-change} | change (`metrics.macro.miou` = 0.8457) |
| **pooled F1** | `2·tp / (2·tp + fp + fn)` from the global counts | change (`metrics.pooled.f1` = 0.8964) |
| **macro F1** | unweighted mean of per-class F1 | change (`metrics.macro.f1` = 0.7962); optical-SAR (`macro_f1` = 0.434161) |

For the change artifact, `metrics.macro.iou` = 0.718 is the **change-class** IoU, while
`metrics.macro.miou` = 0.8457 is the **mean over both classes** — these are different numbers from the
same block, and the headline uses `miou`.

### 10.2 Mean best IoU, mean matched IoU, recall@IoU

| Term | Definition | Where used |
|---|---|---|
| **mean best IoU** | for each record, the maximum IoU over all predicted boxes; then the mean over records | grounding (`mean_best_iou` = 0.2838 / 0.2566 / 0.1215 / 0.0972) |
| **mean matched IoU** | the IoU of the *matched* prediction only, so unmatched records contribute 0 | grounding (`mean_matched_iou`, equal to `mean_best_iou` in all three strategies) |
| **recall@0.10 / 0.25 / 0.50** | fraction of records whose best predicted box reaches that IoU | grounding (`results.*.recall`) |

**Mean best IoU is a max over predictions, so emitting more boxes raises it mechanically.** That is
exactly why the candidate count is recorded (`head_decode.top_k`) and why the decode-matched protocol
(top_k 6 vs the baseline's 5.99) is reported alongside the config default (top_k 20). See §4.2.

### 10.3 The macro-F1 denominator

For a K-class head, `macro_f1 = (1/K) · Σ_c F1_c`. The **denominator is the label space**, not the
number of classes present in the scored split.

- optical-SAR: `K = 19` (`num_classes`), 14 present, 5 absent. Absent classes contribute `0.0`.
  All-19 mean = **0.434161**; present-14 mean = **0.589218**.
- The artifact records `macro_f1_denominator: "all 19 classes (absent classes contribute 0.0)"`,
  `classes_present`, `classes_absent`, and the full 19-term `_per_class_f1` vector so the denominator
  is auditable rather than assumed.

### 10.4 Percentage points vs fractions

Some artifacts record a **fraction** (0.963) and some a **percentage-point delta** (49.5 pp). The VLM
record uses both: `exact_match: 0.963` is a fraction, while `aggregate_test_delta_pp: 49.5` and the
baseline/adapted `46.80 → 96.30 pp` are percentage points. A reader must not confuse `+49.5 pp` with
`+0.495` of a fraction — the former is a 49.5-point move on a 0–100 scale.

### 10.5 ECE and NLL

| Term | Definition | Where used |
|---|---|---|
| **ECE** | expected calibration error: bin-count-sensitive average gap between confidence and accuracy over equal-width bins | calibration (`metrics.ece_before` = 0.013755, `metrics.ece_after` = 0.014929, `n_bins` = 15) |
| **NLL** | mean negative log-likelihood | calibration (`metrics.nll_before` = 0.689741, `metrics.nll_after` = 0.689631) |

**ECE is bin-count sensitive and is not an aggregate score** — recorded verbatim in the artifact's
`reliability_diagram.note`. A smaller ECE is better; here it **increased** (`ece_improvement` =
−0.001174), which is a negative result.

### 10.6 Coverage, fallback rate and ungated accuracy (router)

| Term | Definition | Where used |
|---|---|---|
| **overall ungated accuracy** | task accuracy over all examples, whether or not the confidence gate fired | router (`overall_ungated_accuracy` = 0.965116) |
| **coverage** | fraction of examples the gate accepted | router (`rows[*].coverage`; shipped 0.848837 at threshold 0.70) |
| **covered task accuracy** | task accuracy **among gated-in examples only** | router (`rows[*].covered_task_accuracy`; shipped 0.972603 at 0.70) |
| **fallback rate** | fraction routed to the deterministic lexical fallback | router (`rows[*].fallback_rate`; shipped 0.151163 at 0.70) |

The headline is the **ungated** figure, because a gated accuracy without its coverage is not comparable
across thresholds. That is why the sweep reports both, and why "ungated" is part of the label.

### 10.7 Why the router's `selected` threshold differs from the `shipped` threshold

`select_by: "covered_accuracy"` maximises accuracy among gated-in examples, which is monotone in the
threshold (at 0.76 the covered accuracy is 1.0 on 68 examples). The **shipped** threshold is 0.70
(coverage 0.848837, covered accuracy 0.972603), which is the config default. The `delta_vs_shipped`
block records the difference: `{coverage: -0.0581, covered_task_accuracy: 0.0274}` — i.e. the selected
threshold covers 5.81 points fewer examples for 2.74 points more covered accuracy. The artifact is
explicit that this is *"a justified default, not a calibrated value."*

---

## 11. Evidence index

| Topic | Evidence |
|---|---|
| Verification tool | `release/tools/verify_readme_metrics.py`; committed output `release/tools/readme_metrics_report.txt` |
| Change | `artifacts/change/eval_test/eval_result.json` |
| Grounding (canonical / matched6) | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`, `…/eval_result_matched6.json`; `docs/PHASE7_RESOLUTION_DECISION.md`, `docs/PHASE8_GROUNDING_HEAD_DECISION.md` |
| Optical-SAR | `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`; `docs/PHASE12_115_METRIC_COMPUTED.md`, `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| Change-VQA | `artifacts/change_vqa/run/PROMOTION.json` |
| Router | `artifacts/router/threshold_sweep_val.json`; `docs/PHASE4_ROUTER_REPORT.md` |
| Calibration | `artifacts/calibration_v001.json` |
| VLM | `artifacts/vlm/phase6_closure.json`; `docs/PHASE6_CLOSURE.md` |
| Live validation | the live-validation record and per-case screenshots; `tests/unit/test_frontend_live_wiring.py` (106 passed) |
| Cross-links | [`EVALUATION.md`](EVALUATION.md) (how each number was produced), [`MODELS.md`](MODELS.md) (the artifacts behind each number), [`DATASETS.md`](DATASETS.md) (the corpora), [`TRAINING.md`](TRAINING.md) (how each head was trained), [`LIMITATIONS.md`](LIMITATIONS.md) (the honest catalogue of gaps), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (reproduction guide) |
