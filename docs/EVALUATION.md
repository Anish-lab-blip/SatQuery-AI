# Evaluation

**Status tags used in this document:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED` ·
`NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

This document describes **how** every number in [`BENCHMARKS.md`](BENCHMARKS.md) was produced, and it
enforces the project's evaluation-honesty rules. It is deliberately conservative: a metric that was not
measured is tagged `NOT RUN`; a metric that got **worse** is shown getting worse; a metric whose
acceptance ruling is undecided is shown as `OPEN`.

It is written to be read in full. It is not a summary. Every protocol, threshold, sample count and
artifact key path below was read from a file in the source repository; where the evidence does not
establish something, the text says so explicitly rather than filling the gap.

**Companion documents.**

| Document | What it covers |
|---|---|
| [`BENCHMARKS.md`](BENCHMARKS.md) | The headline metric table, with source artifacts |
| [`LIMITATIONS.md`](LIMITATIONS.md) | An honest catalogue of everything not done or done poorly |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | The reproduction contract, the invariants, the commands |
| [`MODELS.md`](MODELS.md) | The six released artifacts and their backbones |
| [`TRAINING.md`](TRAINING.md) | How each artifact was trained |

---

## 1. Why this document exists, and the strategy it serves

### 1.1 Evaluation-first strategy

The project's master architecture plan (`Implementation and Architecture plan.md`, §2 "Evaluation-First
Strategy") fixes the engineering priority order as:

```
1. Optical-SAR joint reasoning
2. Change analysis
3. Single-image VQA
4. Grounding
5. Captioning
6. Agent/routing/evidence/reliability
```

That is an **engineering priority, not an official score ranking**. The plan is explicit (§2.1) that it
is not a claim about how a competition would weight the tasks. The plan also fixes (§2.2) which metrics
belong to which task family:

| Task family | Plan §2.2 metric set |
|---|---|
| VQA | exact match; normalized exact match; F1 where appropriate |
| Captioning | the benchmark-prescribed metrics; locally additionally BLEU, ROUGE-L, CIDEr, BERTScore |
| Grounding | IoU; Recall@IoU; mAP where applicable |
| Change | precision; recall; F1; IoU; mIoU |
| Change VQA | answer accuracy; semantic match if the benchmark specifies it |
| Optical-SAR | *task-dependent*: classification accuracy/F1, VQA accuracy, region IoU, mask IoU, change metrics |

The plan states a hard prohibition on the optical-SAR row (§2.2): **"Do not invent an official multimodal
metric."** That prohibition is implemented in code, not just documented — see §3.5.

### 1.2 Leakage prevention, the test firewall and the hidden firewall

The plan devotes three sections to the machinery that makes a number trustworthy at all:

- **§36 Data Leakage Prevention** — every sample carries `dataset_id`, `scene_id`, `sample_id`,
  `source_scene`, `geographic_hash`, `acquisition_date`, `sensor`, `sha256`, `split`; splits are assigned
  by **scene**, never by randomly splitting neighbouring patches.
- **§37 Test Set Firewall** — `evaluation/public_test/` is immutable; training code may not import it;
  evaluation code may read it only in *evaluation mode*; **no cache of test answers is permitted**.
- **§38 Hidden Set Firewall** — there must be **no** `hidden_test_mode` in training; the hidden interface
  accepts input imagery and produces the standard result schema, with no knowledge of the hidden answer.

All three are enforced as **mechanisms** rather than conventions (§3.2–§3.4 below). The reason is stated in
the source: *"a rule enforced by convention is a rule that holds until someone is in a hurry"*
(`evaluation/public_test/corpus.py`).

### 1.3 Stop/go gates and the completion checklist

The plan's §72 Stop/Go Gates define the checkpoints a benchmark must pass before it is allowed to
contribute to a final evaluation. **Gate 5** is the one that governs this document:

```
Gate 5 — proceed to final evaluation only when:
  [ ] all prompts frozen
  [ ] all thresholds frozen
  [ ] model revisions frozen
  [ ] config hash recorded
  [ ] public test isolation verified
```

Gate 3 governs grounding (`RemoteCLIP loads` / `VRSBench boxes parse correctly` / `coordinate conversions
tested`). The plan's §79 Completion Checklist enumerates, under `EVALUATION`: manifests, leakage scans,
benchmark adapters, metrics, normalization, immutable public test, reproducible run manifest.

**Status of the checklist for the evaluation family, honestly:**

| Checklist item (§79) | State |
|---|---|
| manifests | `IMPLEMENTED` + `MEASURED` (`evaluation/manifest_freeze.json` exists) |
| leakage scans | `IMPLEMENTED` (`evaluation/leakage.py`) — enforced by raising, not warning |
| benchmark adapters | `IMPLEMENTED`, registry empty at import — adapter-based benchmark runs are `NOT RUN` |
| metrics | `IMPLEMENTED` + `MEASURED` (see §4) |
| normalization | `IMPLEMENTED` (`evaluation/normalize.py`); identity-only, deliberately |
| immutable public test | `IMPLEMENTED` (seal/verify machinery) — the corpus directory is **empty**, reported `available=False` |
| reproducible run manifest | `IMPLEMENTED` (`evaluation/run_manifest.py`) — populated from measurement, never placeholders |

### 1.4 The single most important rule

**Do not fabricate.** A number with no artifact does not appear in these docs. Where evidence is missing,
this document writes `UNKNOWN — not established from the available evidence`.

---

## 2. The eight evaluation-honesty rules, in full

These eight rules are the spine of the document. Each is stated, then given its rationale, then given its
enforcement (by tooling, by test, or by explicit convention), and then — where relevant — a counter-example
showing what would go wrong if it were violated.

### Rule 1 — Evidence before claims

**Rule.** Every reported number has an artifact path. A number with no artifact does not appear in the docs.

**Rationale.** The failure mode this prevents is the *plausible number*: a figure that looks like a result,
appears in a summary, and is reproduced forever after because nobody re-derives it. The project hit this
class of bug repeatedly and records the hits (see `docs/PHASE12_115_METRIC_COMPUTED.md` §8, which documents
an earlier revision that published `2,179 passed` computed as `2,171 + 8` — arithmetic presented as a check,
where the total was never measured).

**Enforcement.** [`../tools/verify_readme_metrics.py`](../tools/verify_readme_metrics.py) walks each quoted
claim to its source artifact and compares at the printed precision. It is committed with its output
(`../tools/readme_metrics_report.txt`) and exits non-zero on any mismatch. See §7.6.

**What it forbids here.** No system-level end-to-end accuracy, no composite score, no "overall" number.
There is no such artifact, so there is no such number (§8).

### Rule 2 — Two protocols are never collapsed

**Rule.** Grounding is reported under the **canonical** protocol **and** the **matched6** protocol. Never
quote one alone.

**Rationale.** The two protocols are different measurement procedures over the same 16,159 VRSBench records,
and they give materially different answers: canonical mean best IoU **0.2838**, matched6 **0.2566**. Quoting
0.2838 alone would be selective; quoting 0.2566 alone would understate the shipped decode. Both are true
statements about different procedures.

**Enforcement.** Both protocols have their own artifact file, both are in the verification tool's claim list
(§7.6), and both appear in the headline table.

**Counter-example.** A reader who saw only `0.2838` would conclude the head is ~11 % better than it is under
the stricter matching protocol. That difference is larger than several of the margins the project treats as
decision-relevant elsewhere.

### Rule 3 — Two test sets are never collapsed

**Rule.** Change-VQA is reported on `test` **and** `test2`.

**Rationale.** The two sets disagree, and they disagree in the direction that flatters a selective report:
`test` accuracy **0.697626** / macro-F1 **0.378373**; `test2` accuracy **0.651469** / macro-F1 **0.372309**.
The `test` numbers are the higher pair.

**Enforcement.** All four values are in the verification tool's 20 claims; the artifact
(`artifacts/change_vqa/run/PROMOTION.json`) records all four under `verification`.

**Counter-example.** Quoting `0.697626` alone overstates accuracy by 4.6 pp against the second test set.

### Rule 4 — `accuracy` never travels without `macro-F1` for imbalanced multi-class heads

**Rule.** For the imbalanced multi-class heads (optical-SAR fusion, change-VQA), the macro-F1 is reported
alongside accuracy, always.

**Rationale.** Optical-SAR fusion reaches **0.931** accuracy with **0.434161** macro-F1. Those two numbers
describe very different things: the model is accurate on frequent classes and weak on rare ones. Accuracy
alone would read as "solved"; macro-F1 alone would read as "broken". Neither reading is correct on its own.

**Enforcement.** The pair is quoted together in every table; the verification tool asserts both.

**A subtlety that is itself documented as a trap.** The macro-F1 denominator is **all 19 class slots**, not
the 14 classes present. Five classes absent from the scored split contribute exactly `0.0`, which pulls the
mean down without describing any prediction the head made. The artifact records
`macro_f1_denominator: "all 19 classes (absent classes contribute 0.0)"`, plus `classes_present` and
`classes_absent`, precisely so the figure cannot be misread. The two denominators give two numbers:
`0.434161` (19 slots) and `0.589218` (14 present classes). Both are reproducible; only the 19-slot figure is
the pre-registered definition. See `docs/PHASE12_115_METRIC_COMPUTED.md` §3.5.

**A correction recorded rather than buried.** That same document records that an earlier revision of §3.5
over-generalised the absent-class argument to all seven zero-valued slots. The honest reading is **both**
statements: part of the low macro-F1 is populational (5 absent classes), *and* there is genuine per-class
failure — classes **5** and **6** are **present** in the scored split and score `0.0` (real total misses).
A class that is absent appears in `classes_absent`; classes 5 and 6 do not. The document states that the
error "was wrong in the flattering direction" and was caught by an independent probe that disagreed.

### Rule 5 — Validation is not test

**Rule.** The router figure is labelled **validation, ungated, n = 86**. The test split was **NOT RUN**.

**Rationale.** The router's `artifacts/router/threshold_sweep_val.json` records `split: "val"`,
`n_val: 86`, `n_val_examples_scored: 86`, `n_test_examples_scored: 0`, and `test_split_touched: false`.
The plan's §59 Testing section sets an engineering acceptance target of `task accuracy ≥ 95 %` measured on
`500 validation queries`, `100 hard negatives`, `50 unsupported queries`. The router corpus has **86** val
examples and **0** hard negatives in val (they are held out to test by design). So the recorded number is
**corpus-limited**: `corpus_limited: true`, with the artifact's own note saying this is *"NOT a calibration"*
and that *"the corpus was NOT padded with generated queries"*.

**Enforcement.** The label "overall **ungated** accuracy" is used everywhere; the artifact note travels with
the number; the verification tool asserts `corpus_limited` and `n_val`.

**Counter-example.** Presenting `0.965116` as a test result would be presenting a number measured on 86
examples — with zero hard negatives — as if it were measured on the plan's 500/100/50 corpus.

### Rule 6 — A negative result stays negative

**Rule.** Calibration ECE **worsened** (`0.013755 → 0.014929`) and is shown worsening. It is retained only
because it is part of the frozen configuration, **not** because it helped.

**Rationale.** The fitted temperature is `T = 0.9772732` and `ece_improvement = −0.001174` — negative. The
NLL improved microscopically (`0.689741 → 0.689631`, `nll_improvement 0.00011`) but the calibration error
metric got worse. Reporting the NLL improvement while omitting the ECE regression would be a selective
reading of the same artifact.

**Enforcement.** Both ECE values are in the verification tool's 20 claims; the tool prints
`(negative => calibration did NOT help)` next to `ece_improvement`; the reliability curve plotted on the
Benchmark page is explicitly labelled as the **pre-scaling** diagram, and the calibrated curve is **not
plotted** at all.

### Rule 7 — `USABLE ≠ ACCEPTED`

**Rule.** The VLM adapter's metrics are real (exact_match **0.963**, F1 **0.96432**); its status is
**ACCEPTANCE-REJECTED**.

**Rationale.** The artifact `artifacts/vlm/phase6_closure.json` states the distinction in its own words:

> *"'Verified' answers: is this artifact the one we trained, and does it work? 'Accepted' answers: did it
> clear the bar predeclared before we looked? Both are true, and they are different questions."*

The aggregate endpoint improved by **+49.5 pp** over the unadapted baseline, yet the run is `REJECTED`
because the pre-registered per-class guardrail (V2) failed on one class. A model can be a working
engineering artifact and a rejected research result at the same time.

**Enforcement.** The verification tool asserts that the artifact's `headline` contains the literal string
`ACCEPTANCE-REJECTED`. The deployed caption/VQA path therefore uses the **unadapted** SmolVLM; the adapter is
attachable only via the `SATQUERY_VLM_ADAPTER` environment variable (resolved in `specialists/vqa/model.py`).

### Rule 8 — No composite / vanity score

**Rule.** There is no single headline accuracy for the system, and none is invented by averaging the
per-task numbers.

**Rationale.** The plan (§63 Metric Normalisation) fixes the aggregation surface as
`aggregate: official_weights: null` "until the organizers publish them", and states that the specification
**explicitly prohibits inventing an official aggregate formula**. Averaging per-task metrics with invented
weights would be exactly that.

**Enforcement.** `evaluation/normalize.py` defines `official_weights: None` and an `aggregate()` function
that **raises** while it is `None`. The evaluation runner *calls* `aggregate()` on every run and records the
refusal in the report's `aggregate_score` block, so the prohibition is exercised rather than merely
documented. See §3.5.

---

## 3. The evaluation architecture

The evaluation subsystem is the single place where a benchmark score could be produced, and therefore the
single place where a fabricated score could be produced. Its design concentrates on making fabrication
structurally impossible. This section walks the machinery.

```
evaluation/
├── __init__.py
├── leakage.py                     # split-by-scene, duplicate rejection, the public-test firewall,
│                                  # the hidden-data guard
├── manifests.py                   # DatasetManifest / SampleRecord / RunManifest
├── manifest_freeze.json           # the frozen dataset manifest
├── prompt_freeze.json             # the frozen prompt set
├── normalize.py                   # §63 normalisation + the aggregate prohibition
├── run_manifest.py                # populates a RunManifest from measurement, never placeholders
├── runner.py                      # manifests → adapters → metrics → normalise → report
├── benchmark_adapters/
│   ├── __init__.py                # registry; empty at import by design
│   ├── base.py                    # BenchmarkAdapter contract + BenchmarkStatus
│   ├── _common.py                 # shared machinery (hashing, provenance, corpus resolution)
│   ├── declared.py                # the four declared benchmarks (data, no load())
│   ├── levir_cd.py                # change
│   ├── vrsbench.py                # grounding
│   ├── bigearthnet_s1.py          # optical-SAR
│   ├── change_vqa.py              # change-VQA
│   ├── fixture_vqa.py             # the one synthetic adapter (must be registered by hand)
│   ├── heldout.py                 # held-out split support
│   └── scorecard.py               # the six-state project scorecard
├── metrics/
│   ├── __init__.py                # narrow re-export surface (pinned by a test)
│   ├── change.py                  # precision / recall / F1 / IoU / mIoU (pooled + macro)
│   ├── grounding.py               # IoU, Recall@IoU, greedy matching, box-convention conversion
│   ├── vqa.py                     # exact match / normalized exact match / token F1 / answer accuracy
│   └── caption.py                 # BLEU / ROUGE-L / CIDEr / BERTScore (capability-gated)
└── public_test/
    ├── __init__.py
    └── corpus.py                  # the immutable, sealable public-test corpus
```

### 3.1 The evaluation runner and its status ladder

`evaluation/runner.py` is the single place a benchmark score is produced. Its status ladder, applied in
order per benchmark (`EvaluationRunner.run_one`), is:

1. **no adapter registered** → `NOT_RUN`, with `detail.registry` recording the registered set and the reason.
2. **corpus unavailable** (`BenchmarkNotAvailableError`) → `NOT_RUN`, with `detail.corpus` carrying the
   exception's context.
3. **`load()` or `scorer()` raised** → `FAILED`, with the traceback attached.
4. **adapter claims availability but yields no samples** → `FAILED` (not a zero-sample success — "an adapter
   that is broken" must not look like "a benchmark that scored nothing").
5. **no scorer supplied** → the runner returns without inventing one.
6. **otherwise** → the adapter's own verdict on its material, via `corpus_kind()`.

The status enum is `BenchmarkStatus` in `evaluation/benchmark_adapters/base.py`:

| Status | Meaning (from `_STATUS_LEGEND`) | `is_scored` |
|---|---|---|
| `NOT_RUN` | No execution took place: no adapter, or corpus absent. No number is reported. | no |
| `FAILED` | Execution was attempted and raised. Traceback attached. Any partial value is not reported. | no |
| `DEGRADED` | Completed against a corpus the adapter does not attest as the benchmark's own (a subset or proxy). The numbers describe that corpus. | no |
| `FIXTURE` | Completed against generated material. The arithmetic is real; the subject is not the benchmark. Never quote as a benchmark score. | no |
| `REAL` | Completed against the benchmark's own corpus, as attested by the adapter. **The only status that yields a benchmark score.** | **yes** |

The critical boundary is in `EvaluationRunner.run`: **only `REAL` results are collected into
`report["real_metrics"]`**. `FIXTURE` and `DEGRADED` results go into their own keys
(`fixture_metrics`, `degraded_metrics`). A caller who wants a benchmark number has to ask for
`real_metrics`, which is empty whenever no real corpus was present.

A design change is recorded in the runner's own docstring: the classification step was originally a two-way
branch on `is_official()`, which made `DEGRADED` **unreachable** — so a real-but-partial corpus was labelled
`FIXTURE`, i.e. called synthetic. `corpus_kind()` now answers the question directly, and its base
implementation reproduces the old two-way result exactly, so no adapter changed meaning when the fix landed.

**The empty registry is deliberate.** `evaluation/benchmark_adapters/__init__.py` states: *"The registry is
empty at import."* Nothing is registered as a side effect of importing the module. `register_default_adapters()`
is the explicit act that wires up the four real-corpus adapters (`levir_cd`, `vrsbench`, `bigearthnet_s1`,
`change_vqa_test`). The synthetic fixture is deliberately **absent** from `DEFAULT_ADAPTER_FACTORIES`:
including it would mean the convenience function quietly added a benchmark whose scores describe a generator.

**Measured fact.** The adapters package docstring records (revised 2026-09-23) that three of the four
corpora are on disk and predate the package:

| Benchmark | Corpus on disk | Size |
|---|---|---|
| `levir_cd` | `data/levir/` | 2,048 test tiles |
| `vrsbench` | `training/data/vrsbench/` | 16,159 eval records |
| `bigearthnet_s1` | `data/bigearthnet_v2/reben/...` | 28,000 S1 patches |
| `change_vqa_test` | `data/cdvqa/annotations/` | 39,686 Test questions |

So *"the gap was never the data. It was the **scoring half**."* Adapter-based benchmark runs remain
`NOT RUN` in this release (see §8); the metric artifacts in §4 were produced by the specialists' own
evaluation scripts, not by the adapter runner.

### 3.2 Leakage prevention

`evaluation/leakage.py` states its rules at the top of the file:

```
1. Split by SCENE, never by sample/tile.
2. No scene may appear in more than one split.
3. Duplicate content (same sha256) is rejected.
4. Public test sets are immutable and off-limits to training code.
5. Hidden data must never influence thresholds, prompts, or routing.
```

and adds the enforcement principle: *"Every function here either returns a clean result or raises. None of
them 'warn and continue' — a leakage warning that scrolls past is indistinguishable from no check."*

**`assign_splits_by_scene(records, *, train_ratio=0.8, val_ratio=0.1, seed=42)`.** This is the function that
prevents the classic remote-sensing leakage bug — slicing a large scene into tiles and then randomly
splitting tiles, so that near-duplicate adjacent tiles leak the test set into training. Implementation
notes recorded in the source:

- Scenes are shuffled with a seeded RNG so the assignment is reproducible.
- Scene **order** is sorted first, making the shuffle independent of input order.
- A scene's records all receive the same split — *"enforced, not hoped for"*.
- Records are **emitted in sorted scene order**, not shuffled order: *"the seed controls which scene goes to
  which split; it must not control the order of the returned list, or downstream code taking `records[:N]`
  would silently depend on the seed."*
- Guards raise `LeakageError` for `train_ratio ∉ (0,1)`, `val_ratio ∉ [0,1)`, and `train_ratio + val_ratio ≥ 1`.

**`audit_manifest(manifest, *, require_all_splits=True)`** performs the full audit and raises `LeakageError`
when a hard rule is broken. It checks: no scene in more than one split; no duplicate `sha256`. Two advisory
warnings are recorded (not raised): a split set that is not exactly `{train, val, test}`; and samples whose
`scene_key` fell all the way back to `sample_id` (no `scene_id`, `geographic_hash` or `source_scene`), for
which *"scene-level isolation is vacuous"*.

**`assert_no_scene_overlap(train, test)`** is the hard assertion that two collections share no scene.

**`PublicTestFirewall(public_test_root)`** blocks training-time reads of the public test tree with two
layers: *path inspection* (any read whose resolved path is under the public test root is refused while the
firewall is armed) and *manifest inspection* (a manifest containing test-split records may not be handed to
a training loop). The firewall is **armed by default**; only the evaluation runner disarms it.

**`assert_no_hidden_access(config)`** guards the hidden-evaluation contract (plan §38). It raises unless
`evaluation.hidden_data_access is False` **and** `evaluation.official_aggregate_weights is None`. The
docstring: *"If someone flips that flag to True, evaluation stops here rather than silently tuning on hidden
material."*

### 3.3 The public-test firewall and the immutable corpus

`evaluation/public_test/corpus.py` implements plan §37 as four mechanisms:

| Plan §37 rule | Mechanism |
|---|---|
| Public test sets are immutable | `PublicTestCorpus.seal()` writes a manifest of every file's `sha256` and byte count; `verify()` recomputes and raises on **any** difference |
| Training code may not import them | a corpus is always bound to a `PublicTestFirewall` root, and `read()` refuses any path that escapes the corpus root |
| Evaluation code may read them only in evaluation mode | `PublicTestCorpus` requires an explicit `EvaluationMode` token; training code does not have one |
| No cache of test answers is permitted | there is **no writer** for gold answers anywhere in the module; `forbid_answer_cache()` makes the absence checkable |

**The `EvaluationMode` token is deliberately not a bool.** The docstring: *"`open_public_test(corpus,
mode=True)` would be far too easy to pass from training code by reflex; requiring an object of this type
means the caller had to go and find the constructor."* `evaluation_mode(reason)` requires a non-empty reason,
which travels into every read's record: *"'Why is this code reading the test set?' is the question the
firewall exists to make answerable."*

**`verify()` distinguishes four deltas** because they mean different things: `changed` (content differs —
the gold answers may no longer hold), `added` (new material appeared), `removed` (material disappeared —
results computed against it are no longer reproducible), and `resized` (content identical but byte count is
not — *"should be impossible and therefore signals a bug"*). A `verify()` that only checked hashes of files
still present would miss `removed`, *"which is the one that silently invalidates a published number."*

**Sealing an empty corpus is refused.** The docstring: *"Sealing an empty corpus would produce a seal that
verifies forever, which reads as 'the test set is intact' when the truth is 'there is no test set'. That is
the exact class of false assurance this module removes."*

**The corpus directory is currently empty** (only a `.gitkeep`). The module reports this as
`available=False`, not as an empty passing test set: *"an empty corpus and a corpus that passed the firewall
must not look the same."* `open_public_test()` raises `PublicTestSealError` when the corpus holds no material.

### 3.4 The run manifest

`evaluation/run_manifest.py` exists because a `RunManifest` was being constructed with four fields left at
their defaults (`code_revision=None`, `environment={}`, `prompt_versions={}`, `started_at=None`) — *"a
manifest in name only: it cannot answer 'which code, on which machine, with which prompts'."*

The rule the module follows is stated at the top:

> **Never fabricate. Record the absence.** Every field here is either a value that was genuinely measured,
> or an explicit statement that it was not available. There is no default that looks like data.

Concretely: an unreadable git commit becomes a `code_revision` string that says so, plus the content
snapshot digest that **is** available; a missing prompt module is reported as `"unavailable"`, not as an
empty string that reads like "no prompts were used"; an environment value that cannot be probed is recorded
as `"unavailable"`. The module's own words: *"A placeholder is worse than a gap, because a gap is visibly a
gap."*

Measured facts recorded by the module:

- `environment_block()` records Python version, implementation, platform, machine, processor, executable,
  cwd, repo root, and — when torch is importable — torch version, CUDA availability, CUDA/cuDNN versions,
  GPU count and GPU names. When torch is absent or broken, every torch key is set to
  `"unavailable (<ExceptionType>)"` rather than omitted, *"because an omitted key is indistinguishable from
  'this run needed no torch'."*
- `prompt_versions()` enumerates the prompt modules (`specialists.vqa.prompts`,
  `specialists.optical_sar.prompts`) rather than globbing them, *"so a specialist that stops exporting a
  version should show up as MISSING, not silently drop out of the manifest."* A prompt change is a behaviour
  change that no config hash captures, which is why it belongs in the manifest.
- `code_revision_identifier()` names a dirty working tree explicitly: *"a commit hash from a dirty tree does
  NOT identify the bytes that ran, and a manifest that omits that fact overstates what it proves."*
- `manifest_completeness()` is a **report, not a gate** — *"this module does not decide whether a manifest is
  good enough, because no quality threshold is defined anywhere in the plan."*
- `ended_at` is **deliberately not defaulted**: *"an unset end time must stay unset rather than becoming
  'now', which would claim the run took zero seconds."*

### 3.5 Metric normalisation and the aggregate prohibition

`evaluation/normalize.py` implements plan §63 literally and is *"deliberately small"*:

- The transform set is **minimal**: the only registered transform is the **identity** map, and only for
  metrics whose native range is already `[0, 1]` — `iou`, `miou`, `f1`, `precision`, `recall`, `accuracy`,
  `map`. Each is listed explicitly in `TRANSFORMS`.
- A metric that is **not** registered **raises** `UnregisteredMetricError`. *"It is never guessed at, never
  clamped, and never inverted."* A lower-is-better metric (e.g. an error rate) is **not** handled, because
  doing so *"would require a bound this module has no authority to choose."*
- A value outside `[0, 1]` after its transform **raises** `NormalisationRangeError`; it is never clamped
  silently.
- `official_weights: None = None` with **no default**. `aggregate()` **raises**
  `OfficialWeightsUnavailableError` while it is `None`.

The runner makes the prohibition **exercised, not merely documented**. `aggregate_or_explain()` calls
`aggregate()` and records the refusal:

```
aggregate_score = {
  "available": False, "score": None, "weights": None,
  "error": "OfficialWeightsUnavailableError",
  "reason": "...",
  "note": "No official aggregate exists. Plan section 63 prohibits inventing an aggregate formula,
           so no single score is reported. Per-metric normalised values remain available above."
}
```

The docstring explains why the call is made rather than short-circuited on the `None` check: *"if someone
later assigns weights without implementing a formula, this call still raises, and the report says so. A bare
`if official_weights is None` would have hidden that."*

`normalise_metrics()` also records failures instead of dropping them: a metric that fails normalisation is
**absent** from `normalized` and **present** in the `normalisation` records with its error, *"so a consumer
cannot mistake 'could not be normalised' for 'was not measured'."* Non-numeric values raise rather than being
coerced: *"a metric whose value is a string is a bug in the adapter, and `float('abc')` failing deep inside a
report writer is a worse way to find out."*

### 3.6 The metric implementations

All metric code is deliberately boring: boolean-array arithmetic, every division guarded, tested against
hand-computed values. The grounding module states the principle: *"Reference values are hand-computed in the
tests rather than taken from a library, because a metric that agrees only with itself proves nothing."*

**`evaluation/metrics/change.py` — change detection.** Pure NumPy; no model, no torch, no dataset. Two
aggregation conventions are computed and labelled:

- **pooled**: sum the confusion counts across every tile, then compute one precision/recall/F1/IoU.
- **macro**: compute per tile, then average.

They differ when tile sizes or change fractions differ and can disagree by several points, so *"rather than
pick one and hope, both are computed and labelled."* Empty-change tiles (all background) are **excluded from
the macro average** — otherwise they contribute a recall of `0.0` *"for a property of the dataset rather
than of the model"* — and are counted separately in `n_images_with_change`. Binarisation uses `>=` rather
than `>`: *"a pixel at exactly the threshold is a change. The asymmetry matters for reproducibility, so it is
pinned by a test."* `miou` averages the change-class IoU and the background IoU.

**`evaluation/metrics/grounding.py` — grounding.** Pure NumPy. Coordinate convention is `[x1, y1, x2, y2]`
in the same frame for prediction and target; internally SatQuery uses normalised 0–1
(`CoordinateSystem.NORMALIZED_0_1`), and VRSBench's 0–100 annotations are converted at the adapter boundary,
**never here**: *"Mixing frames here would produce a plausible-looking IoU that means nothing."* Matching is
**greedy, highest-IoU-first**, not Hungarian — *"greedy is what VRSBench-style referring evaluation does and
it is order-independent because the sort is on the IoU value, not on the prediction order."* Ties break on
`(pred_index, target_index)` so the result is deterministic. A pair where either box has zero area yields
`0.0`, not NaN: *"A degenerate prediction is a wrong prediction, not an undefined one."* Recall is
**macro-averaged over images** (mean of per-image recall), not pooled: *"macro is the convention for
referring evaluation because one crowded image should not dominate the score."* Images with no targets are
excluded from the recall average but still counted in `n_images`. `benchmark_to_normalized(box, scale=100.0)`
is a **named function** rather than an inline division, because *"treating those numbers as pixels or as 0-1
is a silent, catastrophic error."*

**`evaluation/metrics/vqa.py` — VQA / Change-VQA text metrics.** Pure stdlib (`re`, `collections`). The
normalisation rule set is the **VQA-v2** convention, published as an inspectable `NORMALIZATION_RULES` tuple:

```
R1 lowercase the text
R2 fix common contractions      ("dont" -> "don't")
R3 map number words to digits   ("two"  -> "2")
R4 remove articles              ("a", "an", "the")
R5 strip punctuation            (each punctuation char becomes a space; a period or comma
                                 BETWEEN TWO DIGITS is preserved, so "3.14" and "1,000" survive)
R6 collapse whitespace and strip
```

`exact_match` is **raw** string equality (case- and punctuation-sensitive); `normalized_exact_match` applies
`normalize_answer` to both sides first. Both return a float in `{0.0, 1.0}` so they compose with averaging
without a cast. `token_f1` uses the **multiset** intersection, so `"cat cat"` against `"cat"` scores `2/3`,
not `1.0` — *"a set intersection would ignore repetition and inflate the score."* `answer_accuracy` is the
fraction of normalised-exact matches and raises if `preds` and `golds` differ in length, *"because a
mismatch means the caller has mis-aligned questions and answers, and scoring the overlap silently would hide
that."* Degenerate cases are documented and pinned: both sides empty → `1.0`; exactly one side empty → `0.0`.

The module records a judgement call plainly: the rule set is *"a CONVENTION CHOICE. The plan does not
mandate it"*, implemented because *"some documented, deterministic rule set is required for 'normalized
exact match' to mean anything, and VQA-v2 is the standard one."* It says the set *"should be ratified — or
replaced — before these numbers are treated as a benchmark figure."*

**`evaluation/metrics/caption.py` — caption metrics.** Implements the plan §2.2 caption set (BLEU, ROUGE-L,
CIDEr, BERTScore) over pinned third-party packages (`sacrebleu==2.4.3`, `rouge_score==0.1.2`,
`pycocoevalcap==1.2`, `bert-score==0.3.13`). `score_captions()` **raises** for a requested-but-unavailable
metric rather than falling back, *"because a silent fallback is how 'BLEU' ends up meaning 'token F1'."*
Measured capability on the authoring machine (2026-09-23): BLEU available (corpus BLEU 100.0 on identical
input, 41.14 on a near-match); ROUGE-L available (1.0 identical, 0.833 near-match); CIDEr **unavailable**
(no `java` on PATH for the PTB tokenizer); BERTScore **unavailable** (the local HuggingFace cache holds a
corrupt `models--roberta-large` entry whose `config.json` is 0 bytes, which `transformers` prefers over a
download). **No caption score is published anywhere**: *"Producing one requires a corpus of predictions and
references, which is a separate act."*

**The `evaluation.metrics` package surface is deliberately narrow and pinned.** `change` and `grounding` are
imported as submodules, not re-exported flat, because both publish the public names `score_one` and
`score_dataset` — *"a flat re-export would silently shadow one module with the other."* The pin is a test:
`tests/unit/test_vqa_metrics.py::test_package_reexports_the_vqa_public_names` asserts
`set(metrics.__all__) == set(vqa.__all__)` exactly.

### 3.7 The project scorecard

`evaluation/benchmark_adapters/scorecard.py` defines a six-state vocabulary so a reader meets the strongest
evidence first and the two kinds of absence last:

| State | Meaning (from `STATE_MEANINGS`) | Score-bearing? |
|---|---|---|
| `REAL` | ran against the benchmark's own corpus; **the only state that may back a published score** | **yes** |
| `DEGRADED` | ran against a real but partial or proxy corpus; the number describes that material | no |
| `FIXTURE` | ran against self-generated material; proves the machinery, never a benchmark score | no |
| `FAILED` | execution was attempted and raised | no |
| `NOT_RUN` | not attempted: no adapter registered, or deliberately skipped | no |
| `RESOURCE_BLOCKED` | not attemptable here: the immutable evaluation corpus does not exist | no |

`SCORE_BEARING_STATES = frozenset({"REAL"})` — *"Which states may carry a number a reader is allowed to
quote. Exactly one."*

---

## 4. Per-task evaluation protocols

Each subsection states the split, the sample count, the thresholds, the protocol, the decode variants where
they exist, the artifact, and the **exact key paths** at which each quoted number lives.

### 4.1 Change detection — `VERIFIED`

**Artifact.** `artifacts/change/eval_test/eval_result.json`

| Property | Value | Key path |
|---|---|---|
| Split | LEVIR-CD-256 test | `n` |
| n | 2,048 | `n` |
| n with change | 935 | `n_images_with_change` |
| Threshold | 0.50 (frozen in config) | `metrics.threshold` |
| Device | `cuda` | `device` |
| Config hash | `78f1e3700da15aa1` | `config_hash` |
| Checkpoint config hash | `78f1e3700da15aa1`, checked | `checkpoint_config_hash`, `checkpoint_config_hash_checked` |
| Config drift | `false` | `config_drift` |
| Wall time | 55.359 s | `metrics.seconds` |

**Pooled metrics** (sum counts across tiles, then compute) — all under `metrics.pooled.*`:

| Metric | Value | Key path |
|---|---|---|
| IoU (change class) | **0.8122** | `metrics.pooled.iou` |
| mIoU (change + background) | 0.9007 | `metrics.pooled.miou` |
| F1 | **0.8964** | `metrics.pooled.f1` |
| Precision | 0.9195 | `metrics.pooled.precision` |
| Recall | 0.8745 | `metrics.pooled.recall` |
| TP / FP / FN / TN | 5,978,997 / 523,658 / 858,407 / 126,856,666 | `metrics.pooled.tp` … `.tn` |
| n_pixels | 134,217,728 | `metrics.pooled.n_pixels` |

**Macro metrics** (per tile, then average over tiles **with change**) — under `metrics.macro.*`:

| Metric | Value | Key path |
|---|---|---|
| IoU | 0.7180 | `metrics.macro.iou` |
| **mIoU** | **0.8457** | `metrics.macro.miou` |
| F1 | 0.7962 | `metrics.macro.f1` |
| Precision | 0.8506 | `metrics.macro.precision` |
| Recall | 0.7757 | `metrics.macro.recall` |

The macro block carries the same pooled confusion counts (`tp`/`fp`/`fn`/`tn`), because `ChangeReport.macro`
is a `ChangeScores` whose `counts` field is the pooled total — see `evaluation/metrics/change.py`,
`score_dataset`.

**Why both pooled and macro.** The test split is only ≈ 5 % changed pixels — `metrics.mean_change_fraction`
is **0.0509**. The per-image change-fraction distribution (`change_fraction_quantiles`) is
`min 0.0`, `p50 0.0`, `p90 0.197205`, `max 0.684937`. Pooled IoU (0.8122) and macro IoU (0.8457) answer
different questions about that imbalance; the macro figure is the `miou` convention (mean of change-class and
background IoU, per tile, averaged over tiles with change).

**Full confusion counts are shipped** so any metric can be recomputed without re-running the model — see
`evaluation/metrics/change.py::scores_from_counts`.

**Model identity.** The artifact embeds the checkpoint's own config (`checkpoint_embedded_config`):
`encoder: resnet18`, `encoder_channels: [64, 128, 256, 512]`, `frozen_encoder: false`,
`pretrained_used: true`, `sa_mode: PAM`, `width: 128`, `attention_budget_bytes: 268435456`. The evaluation
environment is recorded under `environment`: Python `3.12.13`, torch `2.10.0+cu128`, `cuda_available: true`,
platform `Linux-6.12.90+-x86_64-with-glibc2.35`.

**Released artifact.** `change/head.pt` — 63,231,009 bytes, sha256
`c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa`.

**Why this is the only `VERIFIED` headline.** It is the only task whose artifact is a full-split evaluation
against an immutable public split, at a frozen threshold, with a checked config hash and no drift.

### 4.2 Grounding — `MEASURED`, two protocols × three decode variants

**Artifacts.**

- canonical: `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`
- matched6: `artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json`

**Shared protocol.** Split: VRSBench eval, `n_eval_records` **16,159**. Resolution `224` (frozen;
`resolution_frozen: true` in config). Token grid `grid: 7` (7 × 7 = 49 cells). Head decode
(`head_decode`): `nms_iou: 0.5`, `score_threshold: 0.4`, `top_k: 20` in the canonical run and `top_k: 6` in
the matched6 run — the protocol difference. `frozen_config_evaluation: true`, `limited_run: false`,
`config_drift: false`. Environment: CPU, Python `3.11.16`, torch `2.14.0+cpu`,
`Windows-10-10.0.26200-SP0`.

**Box convention.** VRSBench stores boxes normalised to **0–100**; SatQuery stores **0–1**. The conversion is
a declared config value, `grounding.benchmark_box_scale: 100.0`, and a named function
`benchmark_to_normalized()` — *"so the conversion cannot be applied twice or forgotten."*

**Results — canonical** (`results.*`):

| Decode | mean_best_IoU | Key path | recall@0.10 | recall@0.25 | recall@0.50 | latency ms/img |
|---|---|---|---|---|---|---|
| `head_threshold` (shipped) | **0.2838** | `results.head_threshold.mean_best_iou` | 0.6882 | 0.5047 | **0.2198** | 2.205 |
| `head_argmax` | **0.1215** | `results.head_argmax.mean_best_iou` | 0.3183 | 0.2088 | 0.0795 | 0.655 |
| `zero_shot_matched` (baseline) | **0.0972** | `results.zero_shot_matched.mean_best_iou` | 0.3298 | 0.1188 | 0.0234 | — |

The `recall` sub-dict is keyed by the threshold **as a string** — `"0.10"`, `"0.25"`, `"0.50"` — which is why
the verification tool's dotted-path resolver prefers the **longest** matching key at each step (§7.6). The
recall@0.5 key path is `results.head_threshold.recall.0.50`.

**Results — matched6** (`results.*`):

| Decode | mean_best_IoU | Key path | recall@0.10 | recall@0.25 | recall@0.50 |
|---|---|---|---|---|---|
| `head_threshold` | **0.2566** | `results.head_threshold.mean_best_iou` | 0.6315 | 0.4545 | **0.1938** |
| `head_argmax` | 0.1215 | `results.head_argmax.mean_best_iou` | 0.3183 | 0.2088 | 0.0795 |
| `zero_shot_matched` | 0.0972 | `results.zero_shot_matched.mean_best_iou` | 0.3298 | 0.1188 | 0.0234 |

**Reading.** The head clears the zero-shot baseline, but only the threshold decode is meaningfully above it;
the argmax decode (0.1215) is barely better than zero-shot (0.0972). The absolute level is modest either way:
**grounding is useful, not solved.** The zero-shot baseline reference is also recorded inside both artifacts
as `phase7_reference` (`mean_best_iou: 0.0972`, `recall_at_0.50: 0.0234`, source
`docs/PHASE7_RESOLUTION_DECISION.md`).

**Both protocols are also recorded in a paired analysis** (see §5): the pre-registered resolution experiment
scored **the same 16,159 samples** at both resolutions, which makes the paired test the stronger statistic
because it removes between-object variance.

**Head identity.** `grounding/head.pt` — 12,639,041 bytes, sha256
`93432f7034be91a8ffd9c1a84e3eeec00bed7832c043fe7f83d2be230284c6bb`, over the frozen RemoteCLIP ViT-B/32
encoder (revision `bf1d8a3ccf2d`). Per-cell feature = `concat([patch, text, patch·text, global_pool])` =
`4 × 512 = 2048` (finding P7-1: the transformer width is 768, but `visual.proj` maps to a projected dim of
**512**). Cells are assigned by ground-truth box centre (`decode: cell_relative`). The objectness BCE is
weighted **20×** (`positive_confidence_weight: 20.0`) because only ~1 of 49 cells is positive; unweighted,
*"the optimum is 'no object' everywhere; this weight is what stops that collapse."*

**Two additional pre-registered smoke files** exist for the resolution experiment and are **not** evidence:
`grounding/per_sample_224_smoke.jsonl`, `grounding/per_sample_448_smoke.jsonl`,
`grounding/resolution_experiment_smoke.json`. At n = 12, n = 40 and n = 6 the smoke runs reported
`Recall@0.5 = 0.0000` at **both** resolutions and emitted a degeneracy warning; at full scale the metric is
non-zero (0.0234 / 0.0212), so the note correctly did not fire. The sub-floor runs were never treated as
evidence.

### 4.3 Optical-SAR fusion — `MEASURED`, ruling `OPEN`

**Artifact.** `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`

| Property | Value | Key path |
|---|---|---|
| Tool | `fusion_115_metric` | `tool` |
| Metric | `pre_registered_11.5` | `metric` |
| Definition | fusion-head accuracy and macro-F1 over the 19-class label space on the held-out split | `definition` |
| Split | `test` (held out; never used for selection) | `split` |
| n scored | **4,000** | `n_scored` |
| **Accuracy** | **0.931** | `accuracy` |
| **macro-F1** | **0.434161** | `macro_f1` |
| Loss | 0.254592 | `loss` |
| num_classes | 19 | `num_classes` |
| macro-F1 denominator | all 19 classes (absent classes contribute 0.0) | `macro_f1_denominator` |
| classes present | `[0,2,3,4,5,6,7,8,9,10,12,13,17,18]` | `classes_present` |
| classes absent | `[1,11,14,15,16]` | `classes_absent` |
| Deciding statistic? | **`false`** | `is_deciding_statistic` |
| Head sha256 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` | `head_sha256` |
| Head bytes | 14,427,457 | `head_bytes` |
| Cache | `artifacts/optical_sar/fusion_features/test.npz`, arm A | `cache_path`, `cache_arm` |

**The 19-term per-class F1 vector** (`_per_class_f1`) is shipped so the macro-F1 is auditable:

```
[1.0, 0.0, 0.4280155642023346, 0.5714285714285714, 0.8871595330739299, 0.0, 0.0, 0.8,
 0.928652321630804, 0.9397590361445783, 0.18181818181818182, 0.0, 0.2, 0.4375, 0.0,
 0.0, 0.0, 0.8767123287671232, 0.9980101702409905]
```

**The artifact's own advisory**, verbatim:

> *"This tool reports ONE head's held-out accuracy and macro-F1. It selects no head, ranks nothing and
> compares no arms. Whether this constitutes a Phase 12 pass is the owner's ruling."*

**Two readings, both required.** `docs/PHASE12_115_METRIC_COMPUTED.md` establishes, with four independent
reproductions, that the numbers are correct — and then establishes what they do **not** mean:

- **Reproduced outside the tool** (direct call to `evaluate_fusion_head`): `n=4000, accuracy=0.931,
  macro_f1=0.43416082670034284`.
- **Reproduced without the trainer's helpers at all** — rebuilding the input tensor by hand
  (`concat([optical_gap, sar_gap, joint_gap], axis=1)` then `concat([…, optical_mask, sar_mask], axis=1)`,
  widths `(4000,768)×3 + (4000,12) + (4000,2)` = **2318**) — gives **accuracy 0.931** again. This checks the
  feature *ordering*, which a helper-based re-run would not.
- **Reproduced by a third-party implementation** — `sklearn.metrics.accuracy_score = 0.931`,
  `f1_score(average="macro", labels=range(19), zero_division=0) = 0.43416082670034284`.
- **The accuracy is not a constant predictor.** The test split's majority class holds **2,264 / 4,000 =
  0.566**; the head scores **0.931**.

**The per-class spread is real, in both directions.** Per-class recall on the held-out split (from
`docs/PHASE12_115_METRIC_COMPUTED.md` §3.4): class 18 (n=2264) **0.997**; class 9 (n=246) **0.951**;
class 17 (n=34) **0.941**; class 8 (n=453) **0.905**; class 4 (n=846) 0.809; class 2 (n=63) 0.873;
class 0 (n=13) 1.000; class 13 (n=18) 0.778; class 10 (n=4) 0.500; class 3 (n=39) 0.410; class 12 (n=9)
0.111; class 5 (n=4) **0.000**; class 6 (n=1) **0.000**; class 7 (n=6) 1.000.

Measured test-set label distribution:

```
[13, 0, 63, 39, 846, 4, 1, 6, 453, 246, 4, 0, 9, 18, 0, 0, 0, 34, 2264]
```

- **14 of 19 classes present; five have zero samples.**
- Top-to-bottom ratio **2,264 : 1** (class 18 vs class 6).
- The median of the 14 present classes is **0.6857** against an accuracy of **0.931** — *"the signature of
  prediction dominated by frequent classes."*

**The denominator trap.** `0.434161` and `0.931000` sit next to each other, which invites the reading "the
head is accurate on common classes and catastrophic elsewhere." Two different numbers come from the same
per-class scores:

| Averaged over | Value |
|---|---|
| **all 19 slots** — the pre-registered definition | **0.434161** |
| the 14 present classes only | 0.589218 |

*A reader who computes the second and compares it to the recorded scalar will conclude the recorded figure is
wrong. It is not — it is the 19-slot mean, and the trainer's `_macro_f1` divides by `num_classes` by
construction.* A regression test (`test_macro_f1_averages_over_all_slots_not_present_ones`) pins the
arithmetic.

**The caveat that governs how this number may be used.** The cache metadata records
`label_policy = require_single_label`, `n_skipped_by_policy = 0`. reBEN v2.0 is a **multi-label** corpus while
the frozen head is a **single-label 19-class softmax** trained with `cross_entropy`. The extraction therefore
restricted to single-label patches, which *"preserves the frozen architecture and the 19-class space exactly
[but] changes the **evaluation population**."* Measured context: single-label patches are **17.57 %** of the
corpus (**96,537 of 549,488**), and under this policy the rarest class survives as **1 patch**, a
**59,204 : 1** imbalance.

So this metric **may not** be presented as: a multi-label BigEarthNet/reBEN result; comparable to published
BigEarthNet numbers (almost all multi-label); or a statement about all 19 classes (5 have no test samples).
It **may** be presented as: the pre-registered 11.5 metric, as computed under the single-label extraction
policy the frozen architecture requires, on the held-out split. The `label_policy` choice remains an **open
owner decision**.

**Reproduction and exit codes.** `scripts/eval_fusion_115.py` exits `0` when a metric was computed and `2`
when it refused (missing artifact, empty split, wrong `config_hash`, or a head/cache dimension disagreement —
*"a category error, not a result"*). The tool never reads the validation split when scoring `test`; a test
pins that separation. The trainer's `pre_registered_metric_computed = false` flag is **correct about the
trainer** and is deliberately not "fixed": the trainer fits on `train`/`val` and never opens the test split,
so the held-out split cannot be contaminated by the search over 10 runs, 2 arms and 5 seeds.

**A guard that is load-bearing.** Removing the cache `config_hash` guard makes the tool print
`ACCURACY: 0.000000` for a cache from a different experiment — *"a silent wrong number that looks like a
result."* The guard is pinned by `tests/unit/test_eval_fusion_115.py` (8 tests, 0 deleted, 0 weakened at the
time of that document; §8 of the same document records the module at 10 passed in the suite run).

### 4.4 Change-VQA — `MEASURED`, ruling `OPEN`

**Artifact.** `artifacts/change_vqa/run/PROMOTION.json`

**Two test sets — never collapsed:**

| Split | accuracy | macro-F1 | Key path |
|---|---|---|---|
| `test` | **0.697626367** | **0.378373275** | `verification.test_accuracy`, `verification.test_macro_f1` |
| `test2` | **0.651469262** | **0.372308516** | `verification.test2_accuracy`, `verification.test2_macro_f1` |

The `test` numbers are the higher pair. Both are reported; quoting only `test` would overstate the result.

**Baselines recorded in the same artifact** (`verification`): `global_majority_baseline_test: 0.311546`,
`global_majority_baseline_test2: 0.178728`. Sample counts: `n_scored_test: 39686`,
`n_scored_test2: 31036`. `mask_gain: 0.0`.

**The artifact's own ruling**, verbatim from `verification.metric_ruling`:

> *"OPEN — the plan leaves the accuracy/macro-F1 interpretation owner-gated. No official aggregate metric is
> asserted here."*

**Selection and training provenance** (`identity`): `epoch_selected: 8`, `selected_on: "Val answer accuracy"`,
`val_answer_accuracy: 0.700018`, `stop_reason: "early_stopping"`, `seed: 42`, `config_hash:
78f1e3700da15aa1`, `dataset_id: cdvqa`, `feature_spec: change_feat_v1`,
`change_cache_spec: c801326f85a185f8`, `text_cache_spec: d2801ea1a314354a`,
`preprocessing_version: change_vqa_preproc_v1`.

**Promotion gate evidence** (`artifact` + `source` + `verification`): the head is `artifacts/change_vqa/run/head.pt`,
sha256 `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a`, 5,822,809 bytes, architecture
`change_vqa_head_v1`, **1,453,912 parameters**, `satquery_trained: true`, `eval_mode: true`,
**`non_finite_tensors: 0`**, `weights_modified: false`, `byte_identical_to_source: true`. The hash **agrees
across** `model_metadata.json`, `run_record.json` and `hashes.json (run.checkpoint_sha256)`. The verification
block records `checks_passed: 93`, `checks_failed: 0`, `checks_unverified: 0`.

**Frozen dependency** (`frozen_dependency`): the head's change features are backed by the frozen STANet change
detector `artifacts/change/levir_change_v001/head.pt`, sha256
`c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa`, 63,231,009 bytes,
`verified_byte_exact_vs_local: true`.

**Serving wiring** (`serving_wiring`): the expected path in code is `artifacts/change_vqa/run/head.pt`,
declared in `app/serving.py:76-78` (`CHANGE_VQA_HEAD`) and also defaulted by `scripts/train_change_vqa.py:54`
(`DEFAULT_OUT`) and `scripts/evaluate_change_vqa.py:74` (`DEFAULT_CHECKPOINT`). `code_change_required: false`.

**What promotion does and does not do** (`state`): before promotion the artifact was
`TRAINED_UNVERIFIED`; after promotion it is `PROMOTED`. The artifact's own note: *"Promotion records
provenance and wires the serving path. It does not itself confer VERIFIED status; that is the maintainer's
ruling."*

**The head was trained outside this repository** (Kaggle GPU). The head trains on **cached change + text
features**, not raw imagery. The raw CDVQA loader loads examples but has no training loop of its own, and the
two paths are not conflated. See §8 for what that means for reproduction.

### 4.5 VLM adapter — `MEASURED`, `ACCEPTANCE-REJECTED`

**Artifact.** `artifacts/vlm/phase6_closure.json` (`status: CLOSED`)

**Headline** (`headline`), verbatim: *"Phase 6 is closed. The Run 1 LoRA adapter is promoted to the
production VLM adapter: USABLE and VERIFIED, but ACCEPTANCE-REJECTED."*

**Usable metrics** (`why_usable_verified.adapted_test`) on a **frozen 1,000-question subset**:

| Metric | Value | Key path |
|---|---|---|
| exact_match | **0.963** | `why_usable_verified.adapted_test.exact_match` |
| F1 | **0.96432** | `why_usable_verified.adapted_test.f1` |
| precision | 0.963391 | `why_usable_verified.adapted_test.precision` |
| recall | 0.965251 | `why_usable_verified.adapted_test.recall` |
| n | 1,000 (`n_available: 1000`) | `why_usable_verified.adapted_test.n` |
| confusion | tp 500, fp 19, fn 18, tn 463 | `why_usable_verified.adapted_test.confusion` |

Aggregate test delta: `why_usable_verified.aggregate_test_delta_pp = 49.5` (+49.5 pp over the unadapted
baseline). `why_usable_verified.gate_d_reproduced_run1_adapted_control_exactly: true`.

**Why acceptance was rejected** (`why_acceptance_rejected`), with `decision_split: "test"`:

- `test baseline = 46.80 pp, adapted = 96.30 pp, delta = +49.50 pp; required (V1) >= +5.00 pp` — **V1 passed**.
- `V2 (v002) failed: 1 class(es) lost >= 4 questions with z >= 1.96 on test` — **V2 failed**.

The failing class is **Mixed forest**: `adapted_pp: 87.8788`, `baseline_pp: 100.0`, `drop_pp: 12.1212`,
`lost_questions: 4`, `n_questions: 33`, `z: 2.1335`. Class-level summary: `n_classes_failed: 1`,
`n_classes_held: 5`, `n_classes_improved: 11`, `n_classes_total: 19`.

The artifact records the residual risk explicitly (`residual_risk`): *"The verdict rests on 4 questions in
one class of 33 … With no n >= N floor in V2, a 33-question class can flip the verdict of a run whose
aggregate endpoint improved by 49.5 pp. Reported, not resolved."* It also records why this is not a
split artefact (`why_not_a_split_artefact`): the same class degraded on the val split in run 1
(drop 6.4516 pp, n=31), *"so this is a property of the adapter, not an accident of one subset."*

**The pre-registered thresholds** (`why_acceptance_rejected.thresholds_used`): `accept_min_delta_pp: 5.0`,
`accept_min_delta_pp_ceiling: 2.0`, `ceiling_baseline_pp: 95.0`, `max_class_drop_pp: 1.0` (v001),
`min_class_questions: 20`, `test_val_disagreement_pp: 10.0`, `primary_endpoint: "presence-question normalised
exact-match accuracy"`. The v002 criterion (`v2_criterion`) uses `min_class_drop_questions: 4` and
`class_drop_z: 1.96`, with `se_formula: "sqrt((p_b*(1-p_b) + p_a*(1-p_a)) / n)"` and
`z_formula: "(baseline - adapted) / se"`.

**A declared amendment, recorded honestly.** v002 amends v001's V2 only. The artifact records
`declared_after_first_run: true`, `declared_before_training: false` — i.e. the amendment was declared after
the first run and **not** before training — with the reason that v001's flat 1.0 pp threshold *"fires on 0.20
of a question and sits ~8x below the per-class standard error of the delta (2.6-10.6 pp on run 1)."* The
amendment states its 1.96 SE bar is below the contract's ~3 SE figure, so it is *"not a numerically stricter
bar."* V1/V1'/V3/V4 are unchanged from v001.

**Preserved, immutable verdicts** (`preserved_records`): three records, all `immutable: true` —

| Record | Split | Status | Role |
|---|---|---|---|
| `v001_val_rejected` | val | `REJECTED` | the ORIGINAL pre-registered verdict, preserved verbatim |
| `v002_independent_test_rejected` | test | `REJECTED` | the CONTRACT-FACING verdict |
| `v002_val_accepted_NOT_final` | val | `ACCEPTED` | recorded for completeness ONLY — **not** final acceptance |

The third record is explicitly **not** final: it *"decides on the same val subset that motivated v002, which
§7.4 condition 3 forbids."*

**BERTScore is unavailable and recorded as unavailable**, not as a zero or a substitute:
`thresholds_used.bertscore = {"available": false, "reason": "roberta-large is not in the local HuggingFace
cache; BERTScore cannot be computed offline..."}`. BLEU/ROUGE/BERTScore are excluded for the stated reason
that *"target answers are one token; BLEU/ROUGE are meaningless at that length and BERTScore is unavailable
offline."*

**The forward rule** (`forward_rule`) — *"Any future improved adapter MUST be a new experiment/version. It
MUST NOT rewrite, amend, or supersede Run 1's records."* `run1_is_frozen: true`. New work requires a new
artifact, a new run manifest, a new experiment/version identifier, the predeclared acceptance rule applied
as-is (or a new rule version declared before the run it judges), and its own independent test-split
adjudication.

**Production adapter shape** (`production_adapter`): base `HuggingFaceTB/SmolVLM-500M-Instruct` at revision
`a7da5b986cb5`; `use_status: USABLE_VERIFIED`; `acceptance_status: REJECTED`; `lora_rank: 16`,
`lora_alpha: 32`, `lora_target_module_count: 224`, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`; `trainable_params: 8683520`; `trainable_fraction: 0.01682312`;
`precision_recorded: fp16`. `vision_tower_untouched` records that `trainable_subtrees` is exactly
`{'model.text_model': 8683520}`, so the vision model (86,433,024) and connector (11,796,480) are in
`frozen_params` — *"The contract's vision-tower hazard did not occur."*

**How it is enabled** (`how_enabled`): by environment variable, not a code change —
`SATQUERY_VLM_ADAPTER`, resolved in `specialists/vqa/model.py` (`ADAPTER_ENV_VAR` at line 41; resolution
order explicit arg → env var → none at line 254; attached via `PeftModel.from_pretrained` at line 302).
**The deployed caption/VQA path therefore uses the unadapted SmolVLM.**

**Two known traps recorded in the artifact** (`known_traps`), both about `adapter_sha256` naming two different
values: `training/vlm/artifact.py` computes a **TREE HASH** over the `{relpath: sha256}` weight map
(`5c6b8631…`), while `specialists/vqa/model.py::_adapter_sha256` computes the **FILE** sha256 of
`adapter_model.safetensors` (`07c76a75…`). *"Recomputing one and comparing it to the other yields a false
'artifact was altered' conclusion."* And: *"The promoted adapter is NOT checkpoint-2000. The three weight
files have three different digests: top-level `07c76a75…`, checkpoint-1500 `7273588e…`, checkpoint-2000
`bf249943…` So 'just use the last checkpoint' is not equivalent to this artifact."*

**Verification provenance** (`artifact_verification`): verdict `verified`, `loadable_via_production_path:
true`, `tree_hash_matches: true`, `weights_file_sha256_matches: true`, manifest check clean (14/14 files),
`trainable_params` measured two independent ways (`loaded_minus_frozen_equals_measured: true`). Three
informational findings are recorded (`F2_adapter_sha256_name_collision`, `F3_source_zip_scope`,
`F4_promoted_adapter_is_not_a_checkpoint`). The record was amended on review to add the F4 provenance
finding, *"No verification result changed: verdict remains 'verified'."*

**What closure does not claim** (`what_closure_does_not_claim`): that Run 1 was accepted; that the Mixed
forest regression is resolved; that a new adapter exists or is planned; that the v002 rule or verdict was
altered.

### 4.6 Router — `MEASURED`, `TEST NOT RUN`

**Artifact.** `artifacts/router/threshold_sweep_val.json`

| Property | Value | Key path |
|---|---|---|
| Split scored | `val` | `split` |
| n val | **86** | `n_val` |
| n val scored | 86 | `n_val_examples_scored` |
| n test scored | **0** | `n_test_examples_scored` |
| Test split touched? | **`false`** | `test_split_touched` |
| **Overall ungated accuracy** | **0.965116** | `overall_ungated_accuracy` |
| Corpus limited | `true` | `corpus_limited` |
| Corpus total / groups | 576 / 54 | `corpus_total`, `corpus_groups` |
| Split sizes | train 410 / val 86 / test 80 | `split_sizes` |
| Shipped threshold | 0.7 | `shipped_threshold` |
| Val min per-class support | 8 (caption) | `val_min_support` |
| Hard negatives in val | 0 | `hard_negatives_in_val` |
| Plan min val queries | 500 | `plan_min_val_queries` |
| Plan min hard negatives | 100 | `plan_min_hard_negatives` |

**Val task counts** (`val_task_counts`): caption 8, change 20, grounding 14, optical_sar 10, unsupported 19,
vqa 15.

**The artifact's own note**, verbatim:

> *"corpus-limited: val n=86 vs plan >=500. This is NOT a calibration -- the corpus is synthetic and too
> small (min per-class support 8, caption) and val carries 0 hard negatives (hn_* families are held out to
> TEST by design). Selecting a threshold here yields a justified default, not a calibrated value. The corpus
> was NOT padded with generated queries. Backlog P1-9's 'n=80' is the TEST split; the sweep target is val
> n=86. The test split was NOT touched."*

**The threshold sweep** (`rows`, 50 thresholds from 0.50 to 0.99) reports `coverage`,
`covered_task_accuracy`, `fallback_rate` and `n_covered` at each threshold. The shipped row
(`shipped_row`) is threshold 0.70: coverage 0.848837, covered_task_accuracy 0.972603, fallback_rate 0.151163,
n_covered 73. The row selected by the sweep's own criterion (`select_by: "covered_accuracy"`) is
threshold **0.76**: coverage 0.790698, covered_task_accuracy **1.0**, fallback_rate 0.209302, n_covered 68.
The delta against the shipped threshold (`delta_vs_shipped`) is `coverage: -0.0581`,
`covered_task_accuracy: 0.0274` — i.e. the sweep's selection trades 5.8 pp of coverage for 2.7 pp of covered
accuracy. **The shipped threshold remains 0.70.**

**Adapter identity** (`adapter_encoder`, `adapter_config_hash`): frozen
`sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4`, 22,713,216 parameters, `max_length: 128`;
adapter config hash `615478910dc266bf`.

**Why the number is labelled "ungated".** The plan's §59 engineering acceptance target is
`task accuracy ≥ 95 %` on 500 validation queries / 100 hard negatives / 50 unsupported. The recorded corpus is
86 val examples with **0** hard negatives. A router can reach a high accuracy on easy queries while failing
exactly on the hard-negative families the plan calls out (§5 "Hard negatives … This is important"). The
number is therefore reported as **ungated**, on a **corpus-limited** split, and the **test split is
`NOT RUN`**.

**Released artifact.** `router/adapter.pt` — 211,961 bytes, sha256
`8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c`, architecture
*"task/modality adapter over frozen MiniLM embeddings (~50,822 params)"*. The config records the frozen
encoder's training setup (`router.training`): 60 epochs, batch size 64, learning rate 0.001,
weight decay 0.01, task loss weight 1.0, modality loss weight 0.3, binary loss weight 0.5,
`val_ratio: 0.15`, `hard_negatives_to_test: true` — splits are by **group** (template / hard-negative
family), never by example, so *"hard-negative families are placed in test so their accuracy measures
generalisation rather than memorisation"* (finding F4-3).

### 4.7 Calibration — `MEASURED`, not an improvement

**Artifact.** `artifacts/calibration_v001.json`

| Property | Value | Key path |
|---|---|---|
| Fitted on | `Val` | `provenance.fitted_on` |
| n samples | 16,441 | `provenance.n_samples` |
| Temperature | **0.9772731820958189** | `temperature_scaling.temperature` |
| **ECE before** | **0.013755** | `metrics.ece_before` |
| **ECE after** | **0.014929** | `metrics.ece_after` |
| **ece_improvement** | **−0.001174** | `metrics.ece_improvement` |
| NLL before / after | 0.689741 / 0.689631 | `metrics.nll_before`, `metrics.nll_after` |
| nll_improvement | 0.00011 | `metrics.nll_improvement` |
| Bins / classes | 15 / 19 | `metrics.n_bins`, `metrics.n_classes` |
| Held-out splits excluded | `[Test, Test2]` | `provenance.held_out_splits_excluded` |
| Objective | `mean_negative_log_likelihood` | `provenance.objective` |
| Optimizer | `golden_section_on_log_temperature`, 200 iterations | `provenance.optimizer`, `fit_diagnostics.iterations` |
| Space | `multiclass_logits` | `provenance.space` |
| Scope | the `change_vqa` specialist only | `scope.specialist` |

**Result: calibration did not improve — it moved slightly worse.** `ece_improvement` is negative. The NLL
improved microscopically, but the calibration-error metric got worse, and the honest reading is the ECE one.
The scaling is retained because it is part of the frozen configuration, **not** because it helped.

**Consumer contract** (`consumer_contract`): applied as `sigmoid(logit(z) / T)` for a scalar `z` and
`softmax(logits / T)` for a distribution; class `TemperatureCalibration`, module `evidence.confidence`;
read keys `temperature`, `fitted_on|split`, `artifact`, `n_samples`; resolved via
`load_calibration(config, base_dir='configs')`.

**Reliability diagram** (`reliability_diagram`): 15 equal-width bins over predicted-class confidence, with
per-bin accuracy, confidence, count and gap; `ece: 0.013755` (the **pre-scaling** value). The artifact's own
note: *"Equal-width bins over predicted-class confidence. ECE is bin-count sensitive and is not an aggregate
score."* Two bins (0.0–0.0667, 0.0667–0.1333) have `count: 0` and therefore `null` accuracy/confidence/gap.

**Scope caveat** (`scope.note`): *"This temperature calibrates the R-02 change-VQA head's answer confidence.
Other specialists emit their own raw scores and are unaffected."* So this artifact says nothing about the
calibration of any other specialist.

**Provenance** (`provenance`): the calibration is keyed to the change-VQA head `artifacts/change_vqa/run/head.pt`,
sha256 `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a`, `config_hash: 78f1e3700da15aa1`,
`dataset_id: cdvqa`, `feature_spec: change_feat_v1`. `type_mask_applied: false`.

### 4.8 Captioning — `IMPLEMENTED`, benchmark `NOT RUN`

The caption metric set is implemented and capability-gated (§3.6). **No caption benchmark score is
published** — no caption artifact exists under `artifacts/`, and the module states plainly: *"No caption
score is published anywhere by this change. Producing one requires a corpus of predictions and references,
which is a separate act."* The plan itself records the official caption definitions as `UNVERIFIED`
(`:4231-4237`), and ruling R-16 is open.

So captioning sits at: metrics `IMPLEMENTED`; capability measured (BLEU and ROUGE-L available, CIDEr and
BERTScore unavailable on the authoring machine); **benchmark `NOT RUN`**.

---

## 5. The grounding resolution experiment — a pre-registered rejection

This is the project's model of how a configuration decision should be made, and it is documented in full at
`docs/PHASE7_RESOLUTION_DECISION.md` (status `RESOLVED 2026-09-16`, decision **224 px**).

**The rule, fixed before the result was seen:**

```
448 WINS  if Recall@0.5 improves by >= 0.05 absolute
          OR mean best IoU improves by >= 0.05 absolute
224 WINS  otherwise
INCONCLUSIVE if fewer than 30 samples were scored
```

The artifact records `rule_changed_since_preregistration: false`.

**The run.** Full VRSBench eval split, **16,159 / 16,159 records scored at both resolutions**, Tesla T4,
`--all --device cuda --tag full`.

**The result.**

```
224 WINS
  recall@0.5 gain 448/224 : -0.0022
  bestIoU    gain 448/224 : -0.0147
  latency          ratio  : 1.59x
```

Neither component came close to the +0.05 margin. Both were **negative**.

**Measured detail.**

| Metric | 224 | 448 | delta |
|---|---|---|---|
| token grid | 7 × 7 = 49 | 14 × 14 = 196 | 4.0× tokens |
| attention cost (n²) | 1× | 16× | — |
| with boxes | 16159/16159 | 16159/16159 | — |
| **mean best IoU** | **0.0972** | 0.0825 | **−0.0147** |
| Recall@0.10 | **0.3298** | 0.2599 | **−0.0699** |
| Recall@0.25 | **0.1187** | 0.0944 | **−0.0243** |
| Recall@0.50 | **0.0234** | 0.0212 | **−0.0022** |
| latency mean | **20.0 ms** | 31.8 ms | 1.59× |
| latency p90 | **20.9 ms** | 32.9 ms | 1.57× |
| peak VRAM | **592.1 MB** | 599.8 MB | +7.7 MB |
| wall time | **~8.5 min** | ~11.2 min | 1.32× |

**448 is worse on every quality metric and slower. There is no axis on which it wins.**

**Best-IoU distribution** — the shift is a whole-distribution move toward the zero-overlap bucket, not a tail
effect:

| bucket | 224 | 448 |
|---|---|---|
| 0.00–0.10 | 10,829 | 11,957 |
| 0.10–0.25 | 3,412 | 2,677 |
| 0.25–0.50 | 1,540 | 1,182 |
| 0.50–0.75 | 336 | 307 |
| 0.75–1.01 | 42 | 36 |

**Paired analysis — independent confirmation.** Both resolutions scored the **same 16,159 samples**, so the
paired test removes between-object variance:

```
paired samples     : 16159
mean 224           : 0.0972
mean 448           : 0.0825
mean paired diff   : -0.0147   (95% CI -0.0160 .. -0.0134)
t statistic        : -22.63
CI excludes zero   : True
448 better on      :  1371/16159 ( 8.5%)
448 worse on       :  3372/16159 (20.9%)
identical          : 11416/16159 (70.6%)
```

**The paired test and the pre-registered rule agree.** There is no rule-versus-evidence disagreement to
escalate: both say 224, and the confidence interval excludes zero by a wide margin. The win/loss split is
itself informative: 448 wins on only 8.5 % of records and loses on 20.9 % — *"the finer grid is not merely
neutral, it is **actively harmful** on a fifth of the corpus."*

**The recall ladder, paired** (all three CIs exclude zero): 0.10 → −0.0699; 0.25 → −0.0243; 0.50 → −0.0022.
The gap **narrows as the threshold rises**, which is the signature of a method that cannot reach high IoU
either way.

**Degeneracy notes: none at full scale.** At n=12, n=40 and n=6 the smoke runs reported `Recall@0.5 = 0.0000`
at **both** resolutions and the script emitted a degeneracy warning. At full scale the metric is non-zero
(0.0234 / 0.0212), so the note correctly did not fire. *"The warning was therefore a genuine sub-floor
artefact, and the full run resolved it — which is exactly why the sub-floor runs were never treated as
evidence."*

**Why 448 did not help — the honest reading.** The zero-shot method selects a patch by text similarity and
returns that patch's box. At 224 a box is 1/7 of the image; at 448 it is 1/14. A finer grid is only better if
the target is small and the similarity peak lands on the correct fine cell. Two things work against that
here: the peak is not sharper at 448 (splitting each cell into four gives four chances to pick a wrong
sub-cell, and the similarity field on frozen features is smooth); and Recall@0.10 drops the most (−0.0699) —
*"if finer tokens genuinely localised better, the loosest threshold would benefit most."* The document states
plainly: *"This is the zero-shot baseline's limitation, not a property of RemoteCLIP."*

**What this establishes.** Grounding runs at **224**, frozen in `configs/base.yaml` (`grounding.image_size:
224`, `grounding.resolution_frozen: true`). Peak VRAM for the frozen encoder at 224 is **592 MB** — inside the
ZeroGPU free tier and inside a T4's 15 GB with room for a head and the VLM. Encoder latency at 224 on a T4 is
**20 ms/image** (5× faster than the CPU figure of 97 ms). The 224 localization floor is **1/7 of image width
per token**, documented as a known limitation of the zero-shot method.

**What this does NOT establish.** Whether the zero-shot baseline is good (it is not — mean best IoU 0.0972 and
Recall@0.5 0.0234 are **weak**; this is an ablation floor for the Phase 8 head, not a product). Whether a
trained head has the same resolution sensitivity — a learned head could in principle exploit finer positional
information a cosine-argmax cannot; re-opening the resolution question after Phase 8 would be legitimate
**if** the head's validation curve suggests it, and would be a **new pre-registered experiment, not a silent
retune**. And anything about hidden ISRO/SAC imagery — VRSBench is overhead optical; the hidden set is
Cartosat-2S + RISAT, a different distribution entirely.

**Reproduction.**

```bash
python scripts/exp_grounding_resolution.py \
    --vrsbench  <data-root> \
    --checkpoint <RemoteCLIP-ViT-B-32.pt> \
    --all --device cuda --tag full

python scripts/analyze_grounding_resolution.py --tag _full
```

Artifacts: `per_sample_224_full.jsonl`, `per_sample_448_full.jsonl`, `resolution_experiment_full.json` —
16,159 lines each. Every aggregate in the document is recomputable from the JSONL without re-running the
encoder.

**Action taken** (recorded in the document): `configs/base.yaml` retained `image_size: 224` and set
`resolution_frozen: true`, added the `grounding_head` block, extended `grounding_training`;
`allow_resolution_experiment` was **removed** — *"the experiment is complete, and a flag that says 'not yet
decided' is now false"*; `docs/ARCHITECTURE_FREEZE.md` marked the grounding row `RESOLVED`;
`tests/test_config.py` asserts the freeze fields; `specialists/grounding/head.py` trained at grid 7×7.

---

## 6. Behavioural evaluation (live validation)

Accuracy and behaviour are evaluated **separately**. The per-task protocols in §4 measure whether a model is
*correct*; this section measures whether the **deployed pipeline runs and routes correctly** on unseen
imagery and questions. It is behavioural evidence, and it is **not** an accuracy claim.

### 6.1 What was driven

The deployed stack was driven in a **headed browser** against production
(`https://satquery.pages.dev`), **one upload per case**, with per-case screenshots and recorded run
identifiers. Every case exercised the real path:

```
capabilities → assets → infer   (all on the Render orchestrator origin)
```

with a second `assets` call for the pair tasks. Each case produced a real `run_*` identifier, a live trace
bar, and a per-case screenshot.

**The result.**

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

No run id is shared between passes.

### 6.2 The eight cases

**Phase A** re-runs the six capability cases (regression): vqa, caption, grounding, change, change_vqa,
optical_sar. **Phase B** is the router defect itself: each query uploads **one** asset, which is the exact
condition under which the old router collapsed to `vqa`.

| Case | Query | Expected | Dispatched | Run id (pass 3) |
|---|---|---|---|---|
| A1 | What type of terrain dominates this scene? | vqa | vqa | `run_fef26e91e7e6` |
| A2 | Describe the main visual characteristics of this scene. | caption | caption | `run_96281bdfcc08` |
| A3 | Where are the visible buildings in this image? | grounding | grounding | `run_e49adc8d319f` |
| A4 | What changed between the earlier and later image? | change | change | `run_aedc59cbcdc9` |
| A5 | Did the coastline advance between the two observations? | change_vqa | change_vqa | `run_62ca98d510be` |
| A6 | What land-cover characteristics … optical and SAR …? | optical_sar | optical_sar | `run_beacf6aa4e21` |
| **B1** | **Where are the built-up areas in this image?** | **grounding** | **grounding** | **`run_467ffa406f22`** |
| **B2** | **Where is the new airport?** | **grounding** | **grounding** | **`run_46980ba55c62`** |

**The defect, before and after.** `"Where are the built-up areas in this image?"` with one asset:

- **Before** (HEAD `9d57aed`, replayed through the shipped pre-fix functions):
  `reading=change, temporal=required → dispatched=vqa (wanted=change_vqa, substituted=true)`.
  `\bbuilt\b` matched the temporal regex and `area` matched inside `"areas"`.
- **After** (live): `reading=grounding, temporal=none → run_f0d7a90b5aa1` (pass 1),
  `[grounding] Located 6 candidate region(s) … Highest objectness 0.82.`

`"Where is the new airport?"` with one asset:

- **Before**: `reading=change, temporal=required → dispatched=vqa (wanted=change, substituted=true)`.
- **After** (live): `run_69e38a182a71`, 6 regions, highest objectness 0.83.

**The A5 discriminator note.** The interpretation panel shows the router's **reading**; the task actually
dispatched is the one the server tags on the answer. A5 reads `change` but answers `[change_vqa]` — the
documented quantifier upgrade, not a mismatch. The page's Answer block promises an answer, and the server's
`change` returns a spatial map with no language output; with one asset attached, `change` would instead be
refused outright (*"requires exactly 2 assets"*), so the console avoids asking for a pair-requiring task when
only one asset exists. This is **flagged, not failed**.

**Model-quality note (kept separate from deployment success).** The two verdicts are kept apart:

1. **Deployment/integration: PASS** — the full pipeline works on unseen imagery and questions.
2. **Model quality: MIXED** — caption and grounding are meaningful; change/change_vqa are plausible; VQA is
   weak-but-related (A1 answers "Grassland"); optical-SAR still returns a bare class index
   (`[optical_sar] Fused optical-SAR prediction: class_18 (margin 1.000; optical channels 4/12, SAR channels
   2/2)`), not a human label. The modality accounting in that answer again confirms the right channels reached
   the fusion head.

### 6.3 The three passes

| Pass | Deployed HEAD | Harness | Result |
|---|---|---|---|
| 1 | `ff46eba42b18` + `d413d3672311` | v1 (`fill_input`) | 8/8 |
| 2 | `2d7ae53b482d` | v2 asserting | 8/8 |
| 3 | `2d7ae53b482d` | v2 asserting (pre-discriminator-fix) | 8/8 (recomputed) |

Pass 1 ran against the post-fix frontend and the unchanged live backend (`SatQuery-Backend`
`89d80eaddec5`), with the tunnel agent connected (`agent_id: codespaces-fd1038`). Passes 2 and 3 ran against
the final HEAD `2d7ae53b482d`. Each pass produced fresh run identifiers, none shared with the others.

### 6.4 The harness false-positive bug — recorded in full

This is the most important paragraph in this section, because it is exactly the kind of silent
false-positive an evaluation harness must not have.

**What happened.** Pass 1's harness drove the query box with `fill_input()`. A later re-run attempt failed on
case 1 with `run_id=0002`, `mock_nodes=9`, `answer="No answer yet"`, and only a `capabilities` call — the
**mock path**.

**Root cause.** `fill_input` types with **real CDP key events**, and Chrome **drops synthesized key events
when the browser window does not hold OS focus**. The harness had **no assertion** on the query box, so it
clicked Run with the page's **default query** still in the box — and still recorded a "result". That is a
false pass: the harness dispatched the page's default query, not the case's query, and reported success.

**Measured directly.** With Chrome backgrounded, `press_key("Z")` left `#qtext.value` **unchanged**, while
`type_text("Q")` (CDP `Input.insertText`, which is **not** focus-gated) inserted fine.

**The fix.** The current harness **asserts form state before dispatch**:

- `q_ok` — the query box really holds the intended query (`q_got` records what it actually held);
- `obs_ok` — `#obsTail` reads `ready`;
- `t0_ok` — both frames are attached for pair tasks.

Every case in passes 2 and 3 carries `q_ok: true`, `obs_ok: true`, `t0_ok: true`, `no_mock_nodes: true`, a
real `run_*` id, trace fill 94.4444 %, the Hugging Face link in the DOM, and the `capabilities → assets →
infer` chain on the Render origin.

**Two further harness bugs, both causing false *failures* (not false passes):**

1. The answer `[task]` tag exists only for region tasks (vqa/caption answers are bare).
2. The intent panel is a concatenated string, so `task([a-z_]+)` must be matched **non-greedily** up to
   `modality`.

The harness now computes the dispatched task as `answer_tag` when present, else the reading.

### 6.5 The independent check that pass 1 was NOT infected

The bug in §6.4 raised the obvious question: was pass 1's 8/8 itself a false pass? It was checked, and it is
clean, on three independent grounds:

1. **Its intents are query-specific.** A1 reads `taskvqa…temporalnone`, **not** the default query's
   `taskchange…temporalrequired`. If the default query had been dispatched, the recorded intent would be the
   change intent — it is not.
2. **Its answers embed the query text.** Grounding answers read `[grounding] Located 6 candidate region(s)
   for 'Where are the built-up areas in this image?'` — the query text is inside the answer, so the intended
   query reached the server.
3. **A6's answer proves two files were uploaded** (`optical channels 4/12, SAR channels 2/2`), which the
   default single-asset path could not produce.

So pass 1's 8/8 stands. The failure mode is recorded here anyway, because it is the class of bug that
produces a green result from a red run.

### 6.6 The recompute-verdicts safety property

Pass 3 was launched with the harness build that still carried the two discriminator bugs (§6.4), so its raw
output says `SUMMARY 0/8 PASS`. The verdicts (8/8) are **recomputed from the recorded evidence** by
`recompute_verdicts.py`.

That is the intended safety property: **the recorded evidence (run id, `mock_nodes`, intent, answer,
assertions) is independent of the verdict computation**, so a harness bug can never silently turn a real
failure into a pass, and never costs a re-run to correct. The raw pass-3 records show
`task_dispatched: false` (the buggy check) while `answer_tag_agrees: true`, `no_mock_nodes: true`,
`live_run: true` — the recomputation reads the evidence, not the buggy predicate.

**Liveness notes recorded for anyone reproducing this** (`LIVE_VALIDATION_POSTFIX.md`):

- **`browser-use` block-buffers stdout even when redirected to a file**, so the output file stays at 0 bytes
  until the process exits — indistinguishable from a stall. The harness now forces line-buffering and prints
  `CASE_START <id>` per case.
- **`grep` block-buffers when piped**, so piping the harness through `grep` swallows all output if the
  pipeline is killed. Redirect to a file instead.

### 6.7 Known residuals after the live runs

- `"What is the new runway?"` still routes to `change` rather than `vqa` (the `new`-as-change heuristic fires
  on non-`where` questions). Strictly better than pre-fix, where `new` was unconditionally temporal.
  *"A lexical router cannot cleanly separate 'the new X' from 'what's new'."*
- `"How much built-up area was added?"` now routes to `vqa` (under-trigger) because `built` was dropped from
  the temporal set and `area` no longer matches inside `areas`.
- The old EO pair URLs still answer `200` from Cloudflare's **edge cache** (`CF-Cache-Status: HIT`) although
  the files are deleted — a cache-busted request returns `404`. Nothing references them.
- The Anatomy plate points at the 720×720 variant of an image the recorded run analysed at 730×730. The
  content is identical and the canvas scales it, but the page's *"the ACTUAL analysed image"* wording is very
  slightly loose.

---

## 7. Test-suite results

This section reports the test suites **in full, including the failures**. Nothing is hidden, and no failure is
claimed to be a regression when it is not.

### 7.1 Frontend live-wiring regression suite

`tests/unit/test_frontend_live_wiring.py` — **106 passed** (re-run this session).

The module proves two things that were **broken before the change** (quoted from the module docstring):

- **CORS.** The deployment allowed exactly one origin (`https://satquery.pages.dev`), so the frontend could
  not be driven from a local development server at all: every request from `http://localhost:8080` was
  refused with `Disallowed CORS origin`, and a developer had to deploy to Cloudflare to test a one-line
  JavaScript change. The fix adds explicit localhost origins while keeping production listed and keeping `*`
  rejected.
- **The task vocabulary.** The page's router and the server's `Task` enum must agree. They are two
  independently written vocabularies in two languages, and nothing previously checked that a value the
  frontend would send is a value the server accepts. The module reads **both** and compares them.

It also pins the `chang` word-boundary defect: `\bchang\b` cannot match `"changed"`, so the page's own default
question (*"What changed here?"*) routed to `vqa` instead of the change path. *"That is invisible to any test
that only checks 'is this a valid enum value' — both branches produce a valid value. The test therefore
asserts the ROUTE, not just the validity."*

The tests parse the shipped **source text**, because the frontend is plain ES5 with no build step and no
module exports, so there is no importable symbol; the parsing is anchored on **named tokens** rather than line
numbers, *"so it does not silently pass if the file is restructured."*

The test classes, read from the file, cover: the CORS allowlist; the CORS decision; the frontend↔server task
vocabulary; the default question reaching the change path; a location question **not** being a change
question; the architecture policy not reading a location as a change; the task respecting the pair
requirement; the live client using the real ingestion path; the page loading the live client; the honest
failure contract; the caption no longer calling a real upload illustrative; the frontend sending only the
assets the task requires; descriptive queries routing to caption; optical-SAR input validation; and server
error translation.

### 7.2 Doc/frontend suite

The doc/frontend suite (5 files) — **183 passed**. The five files are:

| File | What it pins |
|---|---|
| `tests/unit/test_frontend_guide_doc.py` | both documents agree on tasks, coordinate systems, evidence types, geometry shapes; no login screen; no secrets in the browser; the confidence-property trap; serialized analyses |
| `tests/unit/test_frontend_live_wiring.py` | see §7.1 |
| `tests/unit/test_api_contract_doc.py` | every JSON block parses and validates; no contract example fabricates an artifact URI; documented task/coordinate-system values are the real ones; every error code documented; the frozen config hash is recorded; the calibration caveat is recorded |
| `tests/unit/test_runbook_doc.py` | the runbook names only public serving entrypoints; quoted artifact sizes match the real files; capability list matches the registry; env vars are the architecture ones; undecided items declared undecided; **no claim that a deployment happened**; verified distinguished from designed |
| `tests/unit/test_deploy_config.py` | the deploy manifest declares the non-registry marker; the validator reports no errors; **the config-hash regression guard**; C-8 constraints (`torch_compile` false, cpu mode required, lazy load, single model cache) |

These suites are **documentation conformance tests**: they fail if a doc drifts from the code it describes.

### 7.3 Full `tests/unit` — the 5–6 environmental/ordering failures, reported honestly

Running the **entire** `tests/unit` suite in the authoring sandbox produces **5–6 failures**. They are
**environmental / ordering** failures, and they are attributable as follows:

| # | Failure | Attribution |
|---|---|---|
| 1–4 | `test_safe_delete_shim` — **4 failures** | the sandbox **delete guard** (environment) |
| 5 | one **ordering flake** in the router route test | test **ordering** (passes in isolation) |
| 6 | one **stale adapter test** (`optical_sar` absent when CROMA unshipped) | stale test — **CROMA is now shipped** |

**These are not regressions, and they are not hidden.** The evidence that they are not regressions is the
re-run: **re-running the affected files together gives 137 passed**.

**The `test_safe_delete_shim` failures in detail.** `tests/unit/test_safe_delete_shim.py` guards two defects
in the WorkBuddy Windows safe-delete shim (`cli/vendor/shim/sitecustomize.py`) that produced **phantom test
failures** in this repository. The module's own docstring records the mechanism:

- **Defect 1 — a verbatim temp path was not recognised as a temp path.** `_path_for_compare` returned
  `normcase(realpath(abspath(path)))`, which preserves the Windows verbatim prefix `\\?\`. `os.path.relpath`
  cannot relate `\\?\c:\...` to `c:\...`, so `_is_under_root` returned False and the OS-temp exemption in
  `_should_bypass_safe_delete` did not apply. Measured before the fix: *plain* temp subdir → bypass `True`;
  *verbatim* temp subdir → bypass `False` (the bug); non-temp subdir → bypass `False`. That is why routine
  pytest `garbage-*` collection — which walks
  `\\?\C:\Users\...\Temp\pytest-of-anish\garbage-*` — reached the bulk guard, tripped `confirmRequired` at 69
  entries against a threshold of 50, and **latched a rejection that then blocked every delete in the
  conversation**.
- **Defect 2 — a successful delete was reported as a failure.** `_platform_trash` raised whenever
  `SHFileOperationW` returned non-zero. Measured on the authoring host with
  `FOF_ALLOWUNDO|FOF_NOCONFIRMATION|FOF_NOERRORUI|FOF_SILENT`, calling shell32 directly: 4 regions × 3
  interleaved rounds returned `2, 2, 2, …` for non-temp and `0, 0, 0` for temp; 6 processes × 20 deletes
  returned `2 (120/120)` non-temp and `0 (120/120)` temp. The target is removed in **every** case and the
  Recycle Bin is populated and active (1,300+ `$I`/`$R` entries), so the return code is **not** a reliable
  "could not delete" signal.

The module states that the code is **intermittent** — *"across pytest invocations the same non-temp delete
sometimes returned 0, and in one run a temp delete returned non-zero. The precise Windows-internal trigger is
**NOT established** and is not claimed here."* This is one place where this document must write
**UNKNOWN — not established from the available evidence** (§9).

The module **skips cleanly** when the shim is absent or disabled: `pytest.importorskip("sitecustomize", ...)`
plus `skipif` markers for the shim helpers and for non-Windows hosts. So on a normal user environment these
tests skip; in the authoring sandbox they reach the guard and 4 of the 7 fail.

**The stale adapter test.** One test asserts `optical_sar` is absent when CROMA is unshipped. CROMA is now
shipped, so the assertion is stale. This is a **stale test**, not a code defect.

**The ordering flake.** One router route test fails only when the full suite runs in a particular order, and
**passes in isolation**. That is the definition of an ordering flake.

**A different environment state recorded earlier.** `docs/PHASE12_115_METRIC_COMPUTED.md` §8 records, for
2026-09-22, a full unit suite result of **2,237 passed, 0 failed (622.68 s)**. That is a real, measured
result in a *different* environment state (the safe-delete shim's guard state and the CROMA shipping status
differ). It is recorded here rather than suppressed, because the two results are both true and the difference
is exactly the environmental/ordering story. The same document records a correction: an earlier revision
claimed `2,179 passed` computed as `2,171 + 8`, *"presented as a check but the total was never measured — it
was inferred from a stale baseline."* The measured figure was 2,237.

**UNKNOWN.** The exact **collected** test count for the current-session full-suite run is
`UNKNOWN — not established from the available evidence`: the record gives the failure classification and the
137-passed re-run, but not a collected total.

### 7.4 Why the failures are not regressions — the 137-passed re-run

Re-running the affected files **together** gives **137 passed**. The logic: if the failures were real
regressions in the code under test, re-running those files together would still fail. They do not — which is
what distinguishes an environmental/ordering failure from a regression. The four shim failures are
*environment* (a sandbox guard), the flake is *ordering* (passes in isolation), and the sixth is a *stale
test* whose premise changed (CROMA shipped).

### 7.5 The evidence-engine purity and determinism tests

`tests/unit/test_evidence_engine.py` (73 test functions) pins the evidence engine's two properties that
matter — **determinism** and **no loss** — plus the confidence honesty rule. The module docstring states the
contract:

> *"determinism — the same claims must produce the same ordered, identified collection even when specialists
> finish in a different order, or when the uuids differ between processes. no loss — no specialist's evidence
> may vanish without being counted, and agreement between specialists must be recorded rather than collapsed
> with one name thrown away. Plus the honesty rule on confidence: an uncalibrated engine must pass the raw
> score through and SAY it is uncalibrated, never manufacture a fitted mapping."*

The core purity claim is `test_aggregate_is_deterministic_across_repeat_calls` (line 132): *"Same input ->
byte-identical output."* It asserts three things — identical `ids()`, identical `evidence_digest()`, and
identical `model_dump()` lists — across two calls with the same input. The determinism family also includes
`test_order_is_independent_of_input_order` (forward vs backward input order give the same digest),
`test_equal_scores_order_deterministically_by_coordinates` (*"Ties must not fall back to insertion order —
that is input-order dependence wearing a disguise"*), `test_ids_are_sequential_and_zero_padded`,
`test_ids_restart_from_one_for_each_aggregation`, and `test_source_results_are_not_mutated` /
`test_result_objects_are_not_mutated`.

On the confidence side, the tests pin that the engine never manufactures a fitted mapping:
`test_no_calibration_passes_raw_through_and_says_so`, `test_uncalibrated_is_never_labelled_as_fitted`,
`test_identity_temperature_is_reported_as_uncalibrated`, `test_engine_without_calibration_does_not_claim_calibration`,
and `test_temperature_scaling_is_deterministic`. The calibration artifact I/O is pinned too:
`test_missing_artifact_degrades_instead_of_failing`, `test_missing_artifact_raises_when_required`,
`test_malformed_artifact_is_not_silently_swallowed`, `test_artifact_without_a_temperature_is_rejected`.

**Environment note carried in the module docstring:** *"The `--basetemp=.pytest_tmp` flag is mandatory (see
the project handoff): the default pytest temp root triggers a sandbox denial on this machine."* The module's
`scratch` fixture is deliberately **not** cleaned up on teardown: *"the sandbox's safe-delete guard rejects
the recursive delete."* See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §9.

### 7.6 The metric-verification tool

[`../tools/verify_readme_metrics.py`](../tools/verify_readme_metrics.py) is the machine check behind
Rule 1. It is **read-only**, walks each quoted claim to its source artifact, and compares at the **printed
precision**. It resolves **nested** artifact keys, including keys that themselves contain dots — the `recall`
dict is keyed `"0.10"/"0.25"/"0.50"` — by preferring the **longest** matching key at each step, *"so a naive
`split('.')` walk would break."*

**20 numeric claims** are checked. They are:

| # | Claim | Artifact | Key path |
|---|---|---|---|
| 1 | change pooled IoU | `artifacts/change/eval_test/eval_result.json` | `metrics.pooled.iou` |
| 2 | change macro IoU | same | `metrics.macro.miou` |
| 3 | change pooled F1 | same | `metrics.pooled.f1` |
| 4 | grounding canonical `head_threshold` mean_best_IoU | `…/eval_result_canonical.json` | `results.head_threshold.mean_best_iou` |
| 5 | grounding canonical `head_threshold` recall@0.5 | same | `results.head_threshold.recall.0.50` |
| 6 | grounding matched6 `head_threshold` mean_best_IoU | `…/eval_result_matched6.json` | `results.head_threshold.mean_best_iou` |
| 7 | grounding matched6 `head_threshold` recall@0.5 | same | `results.head_threshold.recall.0.50` |
| 8 | grounding `head_argmax` mean_best_IoU (canonical) | `…/eval_result_canonical.json` | `results.head_argmax.mean_best_iou` |
| 9 | grounding zero-shot baseline IoU (canonical) | same | `results.zero_shot_matched.mean_best_iou` |
| 10 | optical-SAR accuracy | `…/pre_registered_115_metric.json` | `accuracy` |
| 11 | optical-SAR macro_F1 | same | `macro_f1` |
| 12 | change_vqa test accuracy | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test_accuracy` |
| 13 | change_vqa test macro_F1 | same | `verification.test_macro_f1` |
| 14 | change_vqa test2 accuracy | same | `verification.test2_accuracy` |
| 15 | change_vqa test2 macro_F1 | same | `verification.test2_macro_f1` |
| 16 | router overall ungated accuracy | `artifacts/router/threshold_sweep_val.json` | `overall_ungated_accuracy` |
| 17 | calibration ECE before scaling | `artifacts/calibration_v001.json` | `metrics.ece_before` |
| 18 | calibration ECE after scaling | same | `metrics.ece_after` |
| 19 | VLM adapter exact_match | `artifacts/vlm/phase6_closure.json` | `why_usable_verified.adapted_test.exact_match` |
| 20 | VLM adapter F1 | same | `why_usable_verified.adapted_test.f1` |

It **also asserts statuses** — that the VLM `headline` contains the literal `ACCEPTANCE-REJECTED`; the
router's `corpus_limited` and `n_val`; the calibration `temperature` and `ece_improvement`. It prints
`ALL CLAIMS VERIFIED` and exits 0 only when everything matches.

**Committed output** (`../tools/readme_metrics_report.txt`): all 20 rows `MATCH`, and the status assertions
print:

```
VLM headline contains ACCEPTANCE-REJECTED : True
VLM status                               : CLOSED
router corpus_limited                    : True
router n_val                             : 86
calibration temperature (temperature_scaling.temperature) : 0.9772731820958189
calibration ece_improvement              : -0.001174  (negative => calibration did NOT help)

RESULT: ALL CLAIMS VERIFIED
```

The comparison is **at the printed precision**, which is why e.g. change-VQA `test_accuracy` reads
`0.697626367` in the artifact and `0.697626` in the README, and both are `MATCH`.

---

## 8. What is NOT evaluated — exhaustive

This is the honest ledger. Every item here is something the evaluation does **not** establish. Nothing in
this table is a claim of success.

### 8.1 Benchmarks that do not exist or were not run

| Item | State | Note |
|---|---|---|
| **System-level end-to-end accuracy** | **NOT RUN — none exists** | There is no measured end-to-end benchmark of the full router → specialist → envelope pipeline. **No such number is claimed anywhere, and none is produced by averaging the per-task numbers.** |
| **Router test split** | **NOT RUN** | Only the validation split (n = 86) was scored; `n_test_examples_scored: 0`, `test_split_touched: false`. |
| **Benchmark adapters** | **NOT RUN** | The four real-corpus adapters exist and the corpora are on disk, but adapter-based benchmark runs were not executed. The registry is empty at import. |
| **End-to-end latency benchmark** | **NOT RUN** | Per-specialist latency is recorded incidentally (e.g. grounding `latency_ms_per_image` 2.205 ms for the head; change eval 55.359 s for 2,048 tiles). There is no end-to-end latency benchmark. |
| **Cross-dataset generalisation** | **NOT RUN** | Each specialist is evaluated only on its own training-family test split. No specialist is evaluated on another family's data. |
| **Human evaluation** | **NOT RUN** | No human rating of any output. |
| **Adversarial / robustness evaluation** | **NOT RUN** | The plan's §60 Adversarial Tests enumerate cases (blank image, all-zero image, extremely bright/dark, noise, unsupported TIFF, 10000×10000, missing metadata, wrong modality labels, same image twice, one temporal image, three images, optical/SAR size mismatch). Those are **test-case intentions**, not a run robustness benchmark; no adversarial *evaluation* was scored. |
| **Caption benchmark** | **NOT RUN** | The caption metrics are implemented and capability-gated, but **no caption score is published**; there is no caption artifact. |
| **Public-test evaluation** | **NOT RUN** | The public-test corpus directory is **empty**; `available=False`; `open_public_test()` raises rather than evaluating nothing. |
| **Multi-label BigEarthNet/reBEN evaluation** | **NOT RUN** | The scored subset is single-label; a multi-label evaluation was not performed. |
| **Calibrated reliability curve** | **NOT RUN (not plotted)** | The plotted reliability diagram is the pre-scaling one; the calibrated curve is not plotted. |
| **Semantic match for Change-VQA** | **NOT RUN** | The plan leaves it optional ("if the benchmark specifies it"); it is not attempted. |
| **mAP for grounding** | **NOT RUN** | The plan says "mAP where applicable"; no mAP was computed. |

### 8.2 Things measured but whose acceptance ruling is OPEN or REJECTED

| Item | State | Why it must not be read as a pass |
|---|---|---|
| Optical-SAR fusion | **OPEN** | `is_deciding_statistic: False`; the tool *"selects no head, ranks nothing and compares no arms"*. The metric is single-label-subset and not comparable to published BigEarthNet numbers. |
| Change-VQA | **OPEN** | The artifact says *"the plan leaves the accuracy/macro-F1 interpretation owner-gated. No official aggregate metric is asserted here."* |
| VLM adapter | **ACCEPTANCE-REJECTED** | V2's per-class guardrail failed on one class (Mixed forest, 4 questions of 33, z = 2.1335) even though the aggregate endpoint improved by +49.5 pp. |
| Calibration | **not an improvement** | ECE worsened (0.013755 → 0.014929). |
| Grounding | **MEASURED, protocol-sensitive** | Two protocols and three decode variants; absolute values are protocol-sensitive; the argmax decode (0.1215) is barely above the zero-shot baseline (0.0972). |
| Router | **TEST NOT RUN** | Val-only, n = 86, corpus-limited, 0 hard negatives in val. |

### 8.3 Distribution and generalisation gaps

| Item | State |
|---|---|
| Hidden ISRO/SAC imagery (Cartosat-2S + RISAT) | **NOT RUN** — VRSBench is overhead optical; the hidden set is a different distribution entirely. `docs/PHASE7_RESOLUTION_DECISION.md` says so explicitly. |
| Sensors other than Sentinel-1/2 in the fusion path | **NOT RUN** — CROMA's pretrained architecture is Sentinel-1/Sentinel-2 oriented; the sensor adapter exists but no non-S1/S2 evaluation was scored. |
| Grounding on a trained head at 448 | **NOT RUN** — the resolution decision was made on the zero-shot method; a learned head's resolution sensitivity is unknown and would require a new pre-registered experiment. |
| Any claim about all 19 optical-SAR classes | **NOT RUN** — 5 of 19 have no test samples. |
| Any claim about rare classes in change-VQA | **NOT RUN** — the macro-F1 is low and the per-class breakdown is not published in the release artifacts. |

### 8.4 Evaluation-infrastructure items that are implemented but not exercised end-to-end

| Item | State |
|---|---|
| The benchmark-adapter runner against a real corpus | `IMPLEMENTED` — not run in this release (`NOT RUN`). |
| The public-test seal/verify cycle on a real corpus | `IMPLEMENTED` — no corpus exists to seal; sealing an empty corpus is refused by design. |
| The hidden-compatible evaluation mode (plan §62) | `IMPLEMENTED` as a contract; no hidden imagery was available, so it was not run. |
| The `RESOURCE_BLOCKED` scorecard state | `IMPLEMENTED` — describes the case where the immutable evaluation corpus does not exist. |

### 8.5 Explicit non-claims

- **No claim of state-of-the-art performance** on any benchmark.
- **No claim of production readiness** for model quality.
- **No claim that the trained heads generalise** beyond their training-family test splits.
- **No claim that calibration improves confidence.**
- **No claim that the VLM adapter is accepted** for production use.
- **No system-level accuracy** is claimed anywhere.

---

## 9. What is NOT RUN / OPEN / BLOCKED for this topic

| Item | Status |
|---|---|
| System-level end-to-end benchmark | **NOT RUN — none exists** |
| Router test split | **NOT RUN** |
| Benchmark-adapter runs | **NOT RUN** |
| End-to-end latency benchmark | **NOT RUN** |
| Cross-dataset generalisation | **NOT RUN** |
| Human evaluation | **NOT RUN** |
| Adversarial / robustness evaluation | **NOT RUN** |
| Caption benchmark score | **NOT RUN** |
| Public-test corpus evaluation | **NOT RUN** (corpus empty; reported `available=False`) |
| Multi-label BigEarthNet/reBEN evaluation | **NOT RUN** |
| Optical-SAR acceptance ruling | **OPEN** |
| Change-VQA acceptance ruling | **OPEN** |
| VLM adapter acceptance | **REJECTED** (`ACCEPTANCE-REJECTED`) |
| Calibration | **MEASURED — worse**; retained only because frozen |
| Grounding | **MEASURED — protocol-sensitive**; two protocols, three decode variants |
| The 5–6 full-suite test failures | **environmental / ordering**, not regressions (137 passed on the affected files re-run together) |
| The precise Windows-internal trigger for the safe-delete shim's intermittent return code | `UNKNOWN — not established from the available evidence` (per the shim test module) |
| The exact collected count for the current-session full-suite run | `UNKNOWN — not established from the available evidence` |

---

## 10. Where the evidence lives

| What | Where |
|---|---|
| Change detection metrics | `artifacts/change/eval_test/eval_result.json` |
| Change threshold sweep | `artifacts/change/threshold_sweep_val.json` |
| Grounding metrics (canonical) | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` |
| Grounding metrics (matched6) | `artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json` |
| Grounding resolution experiment | `artifacts/grounding/resolution_experiment.json`, `…_ck.json`, `…_smoke.json`; `docs/PHASE7_RESOLUTION_DECISION.md` |
| Grounding per-sample (smoke only) | `artifacts/grounding/per_sample_224_smoke.jsonl`, `…_448_smoke.jsonl` |
| Optical-SAR pre-registered metric | `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json` |
| Optical-SAR computation narrative | `docs/PHASE12_115_METRIC_COMPUTED.md` |
| Optical-SAR label policy | `docs/PHASE12_LABEL_POLICY_DECISION.md` |
| Optical-SAR decisions | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| Change-VQA promotion | `artifacts/change_vqa/run/PROMOTION.json` |
| VLM closure | `artifacts/vlm/phase6_closure.json`; `docs/PHASE6_CLOSURE.md`, `docs/PHASE6_AUDIT_AND_CONTRACT.md`, `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` |
| VLM adapter verification | `artifacts/vlm/run1_test_recovery/adapter_verification.json`, `…/run_manifest.json`, `…/test_adjudication.json` |
| Router threshold sweep | `artifacts/router/threshold_sweep_val.json` |
| Calibration | `artifacts/calibration_v001.json` |
| Metric verification tool + output | `../tools/verify_readme_metrics.py`, `../tools/readme_metrics_test_report.txt` (see `../tools/readme_metrics_report.txt`) |
| Evaluation subsystem source | `evaluation/` (runner, leakage, manifests, normalize, run_manifest, benchmark_adapters/, metrics/, public_test/) |
| Leakage / firewall tests | `tests/unit/test_evaluation_runner.py`, `tests/leakage/`, `tests/unit/test_manifest_freeze.py`, `tests/unit/test_prompt_freeze.py` |
| Metric tests | `tests/unit/test_change.py`, `test_grounding_metrics.py`, `test_vqa_metrics.py`, `test_caption_metrics.py`, `test_eval_normalize.py`, `test_eval_fusion_115.py` |
| Evidence-engine purity/determinism tests | `tests/unit/test_evidence_engine.py` |
| Frontend live-wiring suite | `tests/unit/test_frontend_live_wiring.py` |
| Doc/frontend suites | `tests/unit/test_frontend_guide_doc.py`, `test_api_contract_doc.py`, `test_runbook_doc.py`, `test_deploy_config.py` |
| Safe-delete shim tests | `tests/unit/test_safe_delete_shim.py` |
| Live validation (3 passes) | `live_validation/LIVE_VALIDATION_POSTFIX.md`, `results_final.json`, `results_pass3.json`, `run_output.txt`, `run_final2.txt`, `run_final3.txt`, `A1…B2*.png` |
| Live validation recomputation | `live_validation/recompute_verdicts.py` |
| Final delivery report (test results §7) | `docs/FINAL_DELIVERY_REPORT.md` |
| Frozen config | `configs/base.yaml` §`evaluation`, §`grounding`, §`grounding_head`, §`croma`, §`fusion` |

**Reproduce the machine check:**

```bash
python release/tools/verify_readme_metrics.py
```

It prints `ALL CLAIMS VERIFIED` and exits 0 only when all 20 numeric claims match and the status assertions
hold. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full reproduction guide, including the
environment traps and the exact commands.
