# Limitations

An honest, exhaustive catalogue of everything SatQuery AI does **not** do, does poorly, or does not
know. Negative results and open items are listed here rather than omitted, because a limitation that
is not written down is a limitation that will be discovered by someone else at the worst moment.

**Status tags:** `OPEN` · `NOT RUN` · `REJECTED` · `DEFERRED` · `BY DESIGN` · `MEASURED`.

**How to read this document.** Limitations are numbered `L-01 …` and grouped by area. Each entry
states the limitation, the measured or observed detail behind it, and the file that records it. Where
a value is a status, it is stated exactly as the project's own records state it — a `REJECTED` is never
softened to "usable", an `OPEN` ruling is never presented as settled, and a validation number is never
promoted to a test result.

**Companion documents.** [`BENCHMARKS.md`](BENCHMARKS.md) and [`EVALUATION.md`](EVALUATION.md) hold
the metric honesty rules; [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) holds the findings behind several
entries here; [`DEPLOYMENT.md`](DEPLOYMENT.md) §8/§13 holds the operational blockers B-07 and B-02;
[`architecture/04-router.md`](architecture/04-router.md) documents the router residuals.

---

## 1. Model quality

### L-01 — Grounding IoU is low in absolute terms (`MEASURED`)

The grounding head's mean best IoU is **0.2838** (canonical protocol) and **0.2566** (matched6
protocol) on VRSBench eval, n = 16,159. The trained head clearly beats the zero-shot baseline
(**0.0972**), but 0.28 is not "solved". Recall@0.5 is only **0.2198** (canonical) / **0.1938**
(matched6).

**Evidence:** `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`,
`eval_result_matched6.json`; `docs/PHASE7_RESOLUTION_DECISION.md`; `docs/FINAL_DELIVERY_TODO.md` §1.6.

### L-02 — Grounding is protocol-sensitive; an absolute value is meaningless without its protocol (`MEASURED`)

The same head reports very different numbers under different decode variants:

| Variant | Protocol | mean best IoU | recall@0.5 |
|---|---|---|---|
| `head_threshold` | canonical | 0.2838 | 0.2198 |
| `head_threshold` | matched6 | 0.2566 | 0.1938 |
| `head_argmax` | canonical | 0.1215 | 0.0795 |
| `zero_shot_matched` | canonical | 0.0972 | 0.0234 |

So a single grounding number quoted alone is misleading: the head/threshold versus head/argmax split
changes IoU by more than a factor of two. **Never quote one without the other.**

**Evidence:** `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` and
`eval_result_matched6.json`; `release/DOCS_STYLE_GUIDE.md` §3.

### L-03 — The grounding head did not beat the zero-shot baseline on the validation curve (`MEASURED`)

The Benchmark page carries this as an honest label: the grounding head did not beat the baseline on
validation (`docs/FINAL_DELIVERY_TODO.md` §4 P6-T01). The head's advantage over zero-shot is a
**test-split** result (0.2838 vs 0.0972), not a validation result.

**Evidence:** `docs/FINAL_DELIVERY_TODO.md` §4 P6-T01; `docs/PHASE7_RESOLUTION_DECISION.md` ("the
Phase 8 head must beat 0.0972 to justify itself").

### L-04 — Optical-SAR accuracy is carried by common classes; macro-F1 is low (`MEASURED`, ruling `OPEN`)

The fusion head scores accuracy **0.931** but macro-F1 **0.434161** on a held-out test split of
n = 4,000 over 19 classes. **5 of the 19 classes are absent in the scored split** (`classes_absent:
[1, 11, 14, 15, 16]`) and contribute **0.0** to macro-F1 by construction
(`macro_f1_denominator: "all 19 classes (absent classes contribute 0.0)"`). The wide accuracy–macro-F1
gap is the signature of class imbalance. **Never quote accuracy without macro-F1.**

**Evidence:** `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`;
`docs/FINAL_DELIVERY_TODO.md` §1.6.

### L-05 — The optical-SAR ruling is `OPEN` (`OPEN`)

`pre_registered_115_metric.json` states plainly: *"This tool reports ONE head's held-out accuracy and
macro-F1. It selects no head, ranks nothing and compares no arms. Whether this constitutes a Phase 12
pass is the owner's ruling."* Phase 12 is **INCOMPLETE**; the pre-registered metric's gate criterion is
owner-gated (`docs/PHASE12_CURRENCY_CORRECTION.md` §0, §3; `docs/STEP7_BACKEND_CHAIN_REPORT.md` §17).

**Evidence:** `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`
(`is_deciding_statistic: false`); `docs/PHASE12_CURRENCY_CORRECTION.md`.

### L-06 — Change-VQA is weak on rare classes, on two test sets (`MEASURED`, ruling `OPEN`)

Change-VQA scores **accuracy 0.697626 / macro-F1 0.378373** on `Test` (n = 39,686) and
**0.651469 / 0.372309** on `Test2` (n = 31,036). The wide accuracy–macroF1 gap is the signature of
class imbalance. The ruling is **`OPEN`** (`docs/PHASE19_FINAL_HARDENING.md` §4.5: "the R-02
macro-F1/accuracy gap is unruled … not this work order's to rule"). Top-3 accuracy is 0.964698. The
head's metadata records `confidence_method: "uncalibrated"` and `test_splits_used: false` on the
training record itself.

**Evidence:** `artifacts/change_vqa/run/PROMOTION.json` (`test_accuracy`, `test_macro_f1`,
`test2_accuracy`); `artifacts/change_vqa/run/model_metadata.json`; `docs/PHASE19_FINAL_HARDENING.md`
§4.5.

### L-07 — VQA is weak-but-related (`MEASURED`)

The live VQA path answers broadly related content rather than a crisp class. Measured live: the case
A1 query answered **"Grassland"** for a scene where a more specific answer was expected
(`LIVE_VALIDATION_POSTFIX.md`, "Model-quality note"; `release/CURRENT_RELEASE_STATE.md` §6). This is a
model-quality limitation, not a deployment fault — the pipeline dispatches and returns a real answer.

**Evidence:** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md`.

### L-08 — Optical-SAR returns a bare class index, not a human label (`MEASURED`)

The live service returns `class_18` rather than a human-readable CLC label. The live answer reads
`[optical_sar] Fused optical-SAR prediction: class_18 (margin 1.000; optical channels 4/12, SAR
channels 2/2)` — the modality accounting confirms the right channels reached the fusion head, but the
answer is not interpretable without a label map.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6;
`.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md`.

### L-09 — Calibration made ECE worse (`MEASURED`)

Temperature scaling moved ECE from **0.013755 → 0.014929** (`ece_improvement: −0.001174` — **worse**)
while improving NLL marginally (0.689741 → 0.689631). It is **retained only because it is part of the
frozen config** — **not** because it helped. The fitted temperature is `0.9772731820958189`, fitted on
the Val split with n = 16,441, `effective: true`, `hit_bound: false`. The path is live
(`core/controller.py` calls `load_calibration(config)` and hands the artifact to `EvidenceEngine`), so
a deployed result carries a calibrated value — but the calibration is **not an improvement** and must
never be described as making confidence "more accurate".

**Evidence:** `artifacts/calibration_v001.json` (`metrics.ece_before`, `metrics.ece_after`,
`fit_diagnostics.temperature`); `docs/STEP7_BACKEND_CHAIN_REPORT.md` §7;
`docs/PHASE19_FINAL_HARDENING.md` §9 (calibration-success correction).

### L-10 — The VLM adapter is `ACCEPTANCE-REJECTED` (`REJECTED`)

The Phase-6 LoRA adapter's metrics are **usable** — test exact-match **0.963**, F1 **0.96432**,
aggregate test delta **+49.5 pp** — and the artifact is `USABLE_VERIFIED`. But its acceptance status is
**`REJECTED`**: v001 rejected it on val, and the independent-test rule v002 rejected it on the test
split (one class, `Mixed forest`, lost 4 questions at z = 2.1335). The closure record keeps the two
questions separate: *"'Verified' answers: is this artifact the one we trained, and does it work?
'Accepted' answers: did it clear the bar predeclared before we looked?"* The deployed caption/VQA path
uses the **unadapted** model by default; the adapter is attached only when `SATQUERY_VLM_ADAPTER` is
set. **USABLE ≠ ACCEPTED.**

**Evidence:** `artifacts/vlm/phase6_closure.json` (`production_adapter.acceptance_status`,
`why_acceptance_rejected`, `what_closure_does_not_claim`);
`docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md`.

### L-11 — The VLM acceptance-rule history is a post-hoc rule change (`MEASURED`, disclosed)

v001 was pre-registered (`declared_before_training: true`) and rejected the run on a per-class guardrail
whose 1.0 pp threshold sits **below the measurement resolution** of the data (0.20 questions at n = 20;
per-class SE 2.6–10.6 pp). v002 was declared **after** run 1 (`declared_before_training: false`) and is
documented as a **relaxation** of V2, justified measurement-theoretically, not by the outcome. The
record keeps v001 and its `REJECTED` verdict verbatim and states that v002 "is not a numerically
stricter bar" than the contract's ~3 SE figure. A reader must treat the acceptance verdict as resting
on 4 questions in one class of 33 — the "unfloored minimum-size exposure" the closure reports but does
not resolve.

**Evidence:** `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §4, §7.1, §7.4;
`artifacts/vlm/phase6_closure.json` (`thresholds_used.amendment`, `residual_risk`).

### L-12 — The router number is a validation result, ungated, n = 86 (`MEASURED`, `TEST NOT RUN`)

Router task accuracy is **0.965116** — measured on **validation**, **ungated**, **n = 86**,
**corpus-limited** (`corpus_total: 576`, `corpus_groups: 54`; `corpus_limited: true`). The **test
split was NOT RUN**. The number is indicative only and must never be quoted as a test result.

**Evidence:** `artifacts/router/router_adapter_v001/metadata.json` (`val_task_accuracy`);
`artifacts/router/threshold_sweep_val.json`; `docs/FINAL_DELIVERY_TODO.md` §1.6.

### L-13 — The router's routing is not perfect (known residuals) (`MEASURED`)

See §2. The router is a lexical/embedding classifier over a small corpus; residual misroutes exist and
are documented rather than hidden.

---

## 2. Router residuals (known misroutes)

### L-14 — *"What is the new runway?"* reads `change`, not `vqa` (`OPEN`)

The `new`-as-change heuristic fires on non-`where` questions. This is "strictly better than pre-fix,
where `new` was unconditionally temporal. A lexical router cannot cleanly separate 'the new X' from
'what's new'."

**Evidence:** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Known residuals");
`release/CURRENT_RELEASE_STATE.md` §6.

### L-15 — *"How much built-up area was added?"* reads `vqa` (under-trigger) (`OPEN`)

`built` was dropped from the temporal set during the B-08 fix and `area` no longer matches inside
`areas`, so a change-style quantifier is not caught and the query under-triggers to `vqa`.

**Evidence:** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Known residuals");
`docs/FINAL_DELIVERY_TODO.md` §5 B-08.

### L-16 — The `interpret()` / `chooseTask()` asymmetry is visually surprising (`RESOLVED`, intentional)

For *"What changed between the earlier and later image?"* with **one** asset attached, the console
**reads** `change` while dispatch correctly falls back to **`change_vqa`**. This is **intentional** —
the reading is asset-count-blind (it describes the question's intent), while dispatch is
asset-count-aware (it respects what can actually be computed with the assets present) — but a reader
who sees the reading panel and the answer disagree may mistake it for a defect.

**Evidence:** [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) §7;
`.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Discriminator note").

---

## 3. Evaluation gaps

### L-17 — No system-level end-to-end benchmark exists (`NOT RUN`)

**No end-to-end accuracy is claimed anywhere.** `STATUS.md` states there is no system-level E2E
benchmark, and the Benchmark page is required to label it `NOT RUN`.

**Evidence:** `docs/FINAL_DELIVERY_TODO.md` §1.7 item 7, §5 B-04; `docs/FINAL_DELIVERY_REPORT.md` §6.

### L-18 — The router test split was not run (`NOT RUN`)

See L-12. The test split exists but was never scored.

**Evidence:** `docs/FINAL_DELIVERY_TODO.md` §1.6; `docs/FINAL_DELIVERY_REPORT.md` §6.

### L-19 — The benchmark adapters were not run (`NOT RUN`)

No benchmark adapter is registered; the registry is empty by construction
(`docs/PHASE19_FINAL_HARDENING.md` §9, "Benchmark pass — Not claimed").

**Evidence:** `docs/PHASE19_FINAL_HARDENING.md` §9; `docs/FINAL_DELIVERY_REPORT.md` §6.

### L-20 — No end-to-end latency benchmark (`NOT RUN`)

Per-specialist latency is recorded only incidentally (e.g. grounding encoder latency 2.205 ms/image at
224 in the canonical eval, 20.0 ms/image on a T4 per `docs/PHASE7_RESOLUTION_DECISION.md`). There is no
benchmark of the deployed request path across the four tiers.

**Evidence:** `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`;
`docs/PHASE7_RESOLUTION_DECISION.md`; [`PERFORMANCE.md`](PERFORMANCE.md).

### L-21 — No cross-dataset generalisation (`NOT RUN`)

Each specialist is evaluated only on its own training-family test split (LEVIR-CD for change,
VRSBench for grounding, a BigEarthNet-family split for fusion, CDVQA for change-VQA). Nothing measures
transfer to a different distribution.

**Evidence:** `docs/PHASE7_RESOLUTION_DECISION.md` ("Anything about hidden ISRO/SAC imagery … a
different distribution entirely"); the per-task evaluation sections of [`EVALUATION.md`](EVALUATION.md).

### L-22 — No human evaluation (`NOT RUN`)

No human study of answer quality, usefulness, or failure modes was performed.

**Evidence:** `docs/FINAL_DELIVERY_REPORT.md` §6 (unverified items); this document is the only
catalogue.

### L-23 — No robustness or adversarial evaluation (`NOT RUN`)

No evaluation of behaviour under adversarial, corrupted, or out-of-distribution inputs. The
change specialist carries invalid-data and registration-quality logic
(`docs/ARCHITECTURE_FREEZE.md` §23 false-change handling), but no robustness *evaluation* exists.

**Evidence:** `docs/FINAL_DELIVERY_REPORT.md` §6.

### L-24 — BigEarthNet label semantics make the local metrics non-comparable (`MEASURED`)

The local BigEarthNet subset is **100 % single-label** against the official **1–11 multi-label**
scheme, so metrics computed on it are **not comparable** to published multi-label numbers.

**Evidence:** [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) §9; `docs/PHASE12_LABEL_POLICY_DECISION.md`.

### L-25 — The reliability diagram shipped is the pre-scaling curve (`MEASURED`)

The shipped reliability curve is the **pre-scaling** curve (labelled as such); the calibrated curve is
not plotted. The caption now reads "Measured" and the bins are transcribed from
`artifacts/calibration_v001.json`, but the remaining section-03 PR curves are still labelled
illustrative.

**Evidence:** `docs/FINAL_DELIVERY_TODO.md` §4 P6-T01 note; `artifacts/calibration_v001.json`.

### L-26 — Statistical significance exists for only one decision (`MEASURED`)

Only the grounding resolution decision (448 vs 224) has a paired test with a confidence interval
(§[`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) §3). Every other per-task number is a point estimate with no
significance test.

**Evidence:** `docs/PHASE7_RESOLUTION_DECISION.md` ("Paired analysis — independent confirmation").

### L-27 — The optical-SAR registry state is `degraded`, and the adapter never advertises `loaded` (`MEASURED`)

The registry resolves `optical_sar` to **`degraded`**, not `available`
(`docs/STEP7_BACKEND_CHAIN_REPORT.md` §17). More generally, the capability adapter derives contract
state from **artifact presence** and therefore **never emits `loaded` or `evicted`** — "a model is
resident" is unknowable without loading one, which the metadata path must not do
(`docs/DEPLOYMENT_ARCHITECTURE.md` §3.3.1). A consumer cannot learn from `/v1/capabilities` whether a
model is actually resident.

**Evidence:** `docs/STEP7_BACKEND_CHAIN_REPORT.md` §17; `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3.1.

### L-28 — The trace's registry block is a snapshot of the *previous* request (`MEASURED`)

`SpecialistRegistry.describe()` is assigned into the trace **immediately after planning** and **before
execution**, so on a cold process the client-visible `built` map is necessarily `{}` — including for
the request that is about to build the specialist. The same query returns two different traces
depending on how many requests the process has already served. The same trace is also internally
inconsistent about which clock it uses (`selected_models` is assigned inside execution, so it *does*
reflect the current request while `built` does not). **Severity: low — observability only.**

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.8 (finding F-19).

### L-29 — The uploaded-asset path surface has historically leaked server-side paths (`RESOLVED`, recorded)

A series of findings (F-13 … F-16) recorded that `trace.inputs`, `trace.steps[PARSE].detail`, and
`evidence[].artifact_ref` / `result.change_map` published server-side filesystem paths to an
unauthenticated client. All were fixed (paths reduced to basenames; refs set to `null` with an explicit
non-retrievable warning; no fabricated `artifact://` URIs). Two consequences remain worth recording:
the contract's §2.4 example once showed a fabricated `artifact://` URI the service cannot emit (now
corrected), and the ruling introduced a **signalling** change (F-16c): on a deployment that configures
`change.artifact_dir`, a no-change run now reports `degraded: true` where it previously reported
`degraded: false`.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.3–§5.6; [`SECURITY.md`](SECURITY.md).

---

## 4. Operational limitations

### L-30 — B-07: transient tunnel gaps (`OPEN`)

A request can hang or return `504` when the tunnel agent is briefly absent. The patch is prepared,
**NOT deployed**.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1; `docs/FINAL_DELIVERY_TODO.md` §5 B-07;
`release/CURRENT_RELEASE_STATE.md` §6.

### L-31 — B-07 root shape: the `auto`-mode fallthrough wastes the wake budget (`MEASURED`)

In `auto` transport mode a tunnel timeout **falls through** to the forward path, which then burns
`wake_timeout_s` (120 s) on a `302` → a worst case of ≈ **249 s** (150 + 120). Measured.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6; [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) §6;
[`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1.

### L-32 — Cold start is tens of seconds (`MEASURED`)

Render's free tier sleeps when idle, and the Codespace may be stopped (idle timeout 30 min). The first
request after idle waits for a wake. Documented, not hidden.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §8; `docs/DEPLOYMENT_TOPOLOGY.md` §2.

### L-33 — B-02: `codespace_name` trailing newline (`OPEN`, cosmetic)

The `/api/health` payload reports the raw `codespace_name` with a trailing `\n`. Cosmetic; the wake
path strips it.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §8.2; `docs/FINAL_DELIVERY_TODO.md` §4 P2-T03.

### L-34 — No database, auth, or queue (`BY DESIGN`)

The gateway is stateless by design. There is no persistence of runs or users, no auth layer, and no
request queue (plan §73/§74; `docs/DEPLOYMENT_ARCHITECTURE.md` §6).

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §2.2, §6.

### L-35 — The deployment repositories are private (`BY DESIGN`)

The three deploy repositories return `404` for an outside audience.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §11.4; `docs/FINAL_DELIVERY_TODO.md` §4 P9-T01.

### L-36 — `deploy/` in the monorepo is stale and untracked (`OPEN` trap)

Edits there do not deploy; it is not the deployed source.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §1.1; `docs/FINAL_DELIVERY_TODO.md` §1.1.

### L-37 — The rate limiter is a fairness control, not a protection control (`BY DESIGN`)

The per-IP limiter keys on the first hop of the client-supplied `X-Forwarded-For`; a caller that varies
the header is never throttled (measured 0/8 throttled with a fresh value per request, versus 5/8
without). It is explicitly **not** a security boundary.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.2; `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §4.1.2.

### L-38 — No APM, distributed tracing, or cost accounting (`BY DESIGN`)

Observability is limited to the health payload and per-run traces. There is no APM, no distributed
tracing across the four tiers, and no cost accounting.

**Evidence:** [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md);
[`OPERATIONS.md`](OPERATIONS.md).

### L-39 — Single-region, no HA (`BY DESIGN`)

One Render service, one Codespace. No redundancy, no failover, no multi-region deployment.

**Evidence:** [`DEPLOYMENT.md`](DEPLOYMENT.md) §13.

### L-40 — A saturated asset store is indistinguishable from a misconfigured one (`OPEN`)

`POST /v1/assets` answers `503` both when the store is unconfigured and when it is full; the response
cannot tell them apart, and the counter that would have separated them was **removed** rather than given
a consumer. Treat the `503` on this route as ambiguous.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.1 (finding F-11).

### L-41 — `change_vqa.artifact_dir` is a configured-but-inert key (`OPEN`, low severity)

`change_vqa.artifact_dir` is advertised in the same `optional_config_keys` table as the two keys that
work, but `specialists/change/vqa_specialist.py` **never reads it** (the attribute occurs exactly once,
as an assignment; the module contains no file-writing code). An operator who sets it receives no
artifacts and **no warning**.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.6 (finding F-17).

### L-42 — The Anatomy page's plate image variant is cosmetic-wrong (`OPEN`, cosmetic)

The Anatomy plate points at the **720×720** variant of an image the recorded run analysed at
**730×730**. The content is identical and the canvas scales it, but the page's "the ACTUAL analysed
image" wording is very slightly loose.

**Evidence:** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Known residuals");
`release/CURRENT_RELEASE_STATE.md` §6.

### L-43 — The `/api/capabilities` deployment block carried stale metadata (`RESOLVED`)

The capabilities `deployment` block once claimed `huggingface-spaces`/`zerogpu`. It was recorded as a
stale-metadata item (`docs/FINAL_DELIVERY_TODO.md` §1.7 item 6) and is reconciled in the shipped
contract; the frozen `configs/deploy.yaml` still describes the superseded target (§5 below).

**Evidence:** `docs/FINAL_DELIVERY_TODO.md` §1.7 item 6; [`DEPLOYMENT.md`](DEPLOYMENT.md) §11.

---

## 5. Packaging and licensing

### L-44 — There is no `LICENSE` file (`OPEN`)

**No `LICENSE` file exists** in the source repository. The README says "add a license file before
public release". A licence must be selected before public release of the code. This is a **release
blocker for the code**, not a model defect.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6 ("**No `LICENSE` file exists** in the monorepo;
README says 'add a license file before public release'"); `release/DOCS_STYLE_GUIDE.md` §3.

### L-45 — Backbones are not redistributed (`BY DESIGN`)

The six trained artifacts are small modules; the backbones (SmolVLM, RemoteCLIP, MiniLM, CROMA,
STANet-style encoder) are fetched from their sources at run time, and their licences are their own.
Nothing here re-licenses a backbone.

**Evidence:** [`MODELS.md`](MODELS.md); `app/serving.py` (`resolve_checkpoint_path`, "offline first");
`docs/ARCHITECTURE_FREEZE.md` §2.

### L-46 — The released weights require their backbones (`MEASURED`)

A consumer of the six artifacts must also fetch the pinned backbones at the exact revisions; an
artifact alone is not runnable. The VLM adapter, for example, requires
`HuggingFaceTB/SmolVLM-500M-Instruct` (revision `a7da5b986cb5`) plus the tokenizer/processor files
listed in `phase6_closure.json`.

**Evidence:** `artifacts/vlm/phase6_closure.json` (`production_adapter.files_required_to_serve_standalone`);
`release/CURRENT_RELEASE_STATE.md` §3.

### L-47 — Artifact-tree bloat (`MEASURED`)

`artifacts/` totals roughly **3.7 GB**, almost all of it caches, duplicates and features rather than
released weights:

| Path | Size | Classification |
|---|---|---|
| `artifacts/grounding/remoteclip_grounding_v001/` | 1.4 GB | ARCHIVED (evidence) |
| `artifacts/grounding/remoteclip_grounding_v001.zip` | 774 MB | **DUPLICATE** of the directory above |
| `artifacts/optical_sar/fusion_head_v001/` | 276 MB | superseded by `fusion_head_production_v001` |
| `artifacts/optical_sar/fusion_features/` | 231 MB | reproducible cache |
| `artifacts/optical_sar/fusion_features_armB/` | 231 MB | reproducible cache |
| `artifacts/change/levir_change_cpu_probe_v001/` | 241 MB | **DUPLICATE** probe of `levir_change_v001` |
| `artifacts/change/levir_change_v001/` | 241 MB | ARCHIVED |
| `artifacts/phase12_selection/*.jsonl` | ~235 MB | data-selection manifests (seeds) |

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §3.

### L-48 — The VLM adapter is not committed and lives under `.scratch/` (`MEASURED`)

The adapter is **not** committed (`.gitignore` excludes `artifacts/`, `checkpoints/` and
`*.safetensors`) and its canonical path is under `.scratch/`, which a future cleanup could remove. The
closure record states that moving it to a non-scratch location is "a reasonable follow-up, not a
closure requirement", and that moving it would make the recorded evidence stale.

**Evidence:** `artifacts/vlm/phase6_closure.json` (`production_adapter.reconstruction.why_not_moved`).

### L-49 — The frozen `configs/deploy.yaml` describes a target that does not exist (`OPEN` paperwork)

`configs/deploy.yaml` still declares `platform: huggingface-spaces`, `sdk: gradio`, `zerogpu: true`,
and the `gpu_duration_*` values. It is inert (`registry: false`), never loaded by `core/config.py`, and
cannot be edited without either failing `scripts/validate_deploy_config.py` or moving `Config.hash`. It
is deliberately left undisturbed, but a reader who finds it will reasonably think the project targets a
Gradio ZeroGPU Space.

**Evidence:** `configs/deploy.yaml`; `docs/DEPLOYMENT_DECISION.md` §4; [`DEPLOYMENT.md`](DEPLOYMENT.md) §11.

---

## 6. Documentation caveats

### L-50 — `docs/FINAL_DELIVERY_REPORT.md` §6 is stale (`SUPERSEDED`)

It still lists the bundled EO change pair as **DEGRADED** (726² vs 736² → shape error) and **B-01** as
**BLOCKED**. Both were resolved on 2026-09-25: the EO pair is now same-shape (both 720×720, new `-720`
URLs) → **RESOLVED**; the Hugging Face link is live on all 11 pages → **B-01 CLOSED**.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6 ("Documentation that this sprint supersedes");
`docs/FINAL_DELIVERY_REPORT.md` §6.

### L-51 — The monorepo `README.md` was materially stale (`SUPERSEDED`)

It described a hermetic frontend, an in-progress Render/Codespace, a `/v1/*` contract, omitted the
tunnel, and pointed at the stale `deploy/`. Superseded by this release's README.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6.

### L-52 — The `hf/` docs were stale (`SUPERSEDED`)

`hf/SETUP.md` and `hf/README.md` asserted the project owns no weights and has no HF credentials. Both
were false at release time.

**Evidence:** `release/CURRENT_RELEASE_STATE.md` §6.

### L-53 — Stale-negative documentation is a structural hazard (`MEASURED`)

Three documents written after the Phase-12 A/B experiment still described it as **un-run**, even though
it had completed 34 hours earlier. The lesson recorded: "a stale negative claim is more dangerous than
a stale positive one … 'This has never been executed' is contradicted by *nothing* — no test fails, no
hash moves, no invariant breaks." The existing conformance tests check that documented things *exist*;
**no test can check that a documented absence is still absent**.

**Evidence:** `docs/PHASE12_CURRENCY_CORRECTION.md` §0, §6.

### L-54 — The original master plan describes a superseded deployment (`SUPERSEDED`)

The master plan specifies a Gradio GUI + HF Space + ZeroGPU + Railway, and a single-image workflow set.
The shipped system is a static frontend + Render + Codespace tunnel, serving JSON, with the change-VQA
and optical-SAR capabilities added later. The plan is a design document, not a description of the
shipped system.

**Evidence:** `docs/MASTER_ARCHITECTURE_PLAN.md`; `docs/DEPLOYMENT_DECISION.md` §4;
[`DEPLOYMENT.md`](DEPLOYMENT.md) §11.

### L-55 — `mask_ref` was wrongly listed as a documented path surface (`CORRECTED`)

An audit note listed `Region.mask_ref` as a deliberate documented path surface; that was wrong —
`core/schemas.py` is a bare `str | None = None` with no description, and `grep -r mask_ref docs/` finds
nothing. Recorded so the correction is not lost.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.5 (note on `mask_ref`).

### L-56 — Documentation-validated-by-execution found nine factual errors (`MEASURED`)

Validating every JSON example against the real Pydantic models and every runbook claim against the
repository caught nine factual errors proof-reading had missed — including `CoordinateSystem`
documented as `normalized`/`geographic` when the real values are `normalized_0_1`/`geo`, and `Box`
documented with nested geometry when the real model is flat. The checks are now permanent tests (48
assertions). This is a caveat about how much confidence a *document* can carry: prose is not validated
by these tests.

**Evidence:** `docs/PHASE19_FINAL_HARDENING.md` §3.6; `docs/STEP8_FINAL_CONFORMANCE_AUDIT.md` §13.

---

## 7. Interface, contract, and data-surface limitations

### L-57 — The contract promises forward-compatible reading; the schema forbids it (`OPEN`, contract contradiction)

`docs/API_CONTRACT.md` §1.1 says a consumer **"MUST tolerate unknown fields on read (forward
compatibility)"**, and §1 says additive changes do not bump the version — which only works if unknown
fields are ignorable. But `ResultEnvelope`, `SpecialistResult` and `ExecutionTrace` are all
`extra="forbid"`, so `model_validate` **rejects** a body carrying an unknown key. Both halves are
load-bearing and cannot both hold. The contract's own authority clause makes the **code** authoritative,
so the document's bullet is the inaccurate half — but the document was **not** corrected (a doc edit is
a reviewable change), so a reader of the contract is still told the wrong thing.

**Evidence:** `docs/STEP7_BACKEND_CHAIN_REPORT.md` §15 (finding C-2);
[`architecture/08-api-contract.md`](architecture/08-api-contract.md).

### L-58 — The capability adapter can never report `loaded` or `evicted` (`MEASURED`)

Because the metadata path must not construct a model, `app/deployment.py` derives contract state from
artifact **presence** and reconstructs the registry's word from that state — the inverse direction. One
observable consequence: **`loaded` and `degraded` are never emitted**, since "a model is resident" is
unknowable without loading one, and `evicted` is a runtime model-cache fact no static inspection can
observe. A consumer therefore cannot learn from `/v1/capabilities` whether a model is actually
resident.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3.1;
`docs/STEP8_FINAL_CONFORMANCE_AUDIT.md` §3 (finding C-4).

### L-59 — Two subsystems use overlapping words with different meanings (`MEASURED`)

To the registry, `unavailable` means "no builder could be constructed"; to the contract it means
"present but broken, and **therefore a defect**". Transcribing one into the other without noticing
"would turn a deployment gap into a reported defect, or vice versa". The registry's `available` is not a
contract state at all, and `degraded` is *both* a whole-service status and a capability state — a reader
cannot tell from the word alone which is meant.

**Evidence:** `docs/STEP7_BACKEND_CHAIN_REPORT.md` §10 (findings H-2, M-2).

### L-60 — The capability source of truth was historically two producers (`RESOLVED`, recorded)

Before the owner ruling, `AnalysisController.health()` enumerated **6** capabilities from the registry
while the served `describe_deployment()` enumerated **2** from two `Path.exists()` calls. The registry
is now authoritative and `describe_deployment()` delegates to the single adapter — but the historical
divergence is recorded because "any table that exists in two places will drift, and this one already
had".

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3.1;
`docs/STEP7_BACKEND_CHAIN_REPORT.md` §10, §16 (finding M-1).

### L-61 — `mask_ref` is an undocumented optional field (`OPEN`, low)

`core/schemas.py` declares `mask_ref` as a bare `str | None = None` with no description, and
`grep -r mask_ref docs/` finds nothing. An earlier audit note wrongly listed it as a deliberate
documented path surface; that was corrected. It remains undocumented.

**Evidence:** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.5 (note on `mask_ref`).

### L-62 — VRSBench box coordinates are normalised to 0–100 (`MEASURED`)

VRSBench's repository notes that provided evaluation box coordinates are normalized to **0–100**, so the
evaluator adapter must explicitly convert between that convention and the project's internal **0–1**
representation rather than quietly treating the numbers as pixels. A consumer that reads a grounding box
without the coordinate-system field will misinterpret it.

**Evidence:** `docs/MASTER_ARCHITECTURE_PLAN.md` §13 (grounding head / coordinate convention);
`docs/STEP7_BACKEND_CHAIN_REPORT.md` §4 (`CoordinateSystem` is exactly `{normalized_0_1, pixel, geo}`).

### L-63 — The gateway's rate-limit and size-limit values are implementation choices, not plan facts (`OPEN`)

The plan specifies none of them. The defaults in `GatewayConfig` are choices the implementation made,
and the maintainer is asked to confirm them, because they bound one client's share of the daily budget.
No rate-limit value is specified by the plan.

**Evidence:** `docs/PHASE19_FINAL_HARDENING.md` §4.4; `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §4.1.

### L-64 — `POST /v1/assets` was not in the original plan (`RESOLVED`, historically undefined)

The plan fixes three endpoints, yet `AnalysisRequest.assets` is a list of *handles*, which requires a
fourth. `docs/API_CONTRACT.md` §2.5 records the gap; the endpoint is now implemented, but the
fourth-endpoint decision was historically open and the gateway once answered `501` for it.

**Evidence:** `docs/PHASE19_FINAL_HARDENING.md` §4.2;
`docs/STEP7_BACKEND_CHAIN_REPORT.md` §16.

---

## 8. Dataset and corpus limitations

### L-65 — The change model is evaluated only on LEVIR-CD-256 (`MEASURED`)

The change head's headline (pooled IoU 0.8122 / macro IoU 0.8457 / pooled F1 0.8964, n = 2048,
threshold 0.5) is measured on **LEVIR-CD-256** test. Nothing measures transfer to another change
dataset or to a different sensor pair.

**Evidence:** `artifacts/change/eval_test/eval_result.json`; [`EVALUATION.md`](EVALUATION.md).

### L-66 — The change evaluation records a large pixel imbalance (`MEASURED`)

The change test split is heavily imbalanced: pooled counts record `tp = 5,978,997`, `fp = 523,658`,
`fn = 858,407`, `tn = 126,856,666` over `n_pixels = 134,217,728`, with a mean change fraction of
**0.0509** and only **935 of 2,048** images containing change. A pooled IoU on a 5 % positive pixel rate
is not the same statistic as a balanced one, and the macro/pooled split (macro IoU 0.718 vs pooled IoU
0.8122) reflects that.

**Evidence:** `artifacts/change/eval_test/eval_result.json` (`metrics.pooled`, `metrics.macro`,
`mean_change_fraction`, `n_images_with_change`).

### L-67 — The router corpus is small and corpus-limited (`MEASURED`)

The router's training/evaluation corpus is **576** queries in **54** groups (`corpus_total: 576`,
`corpus_groups: 54`), with a by-task distribution of caption 91, change 115, grounding 128, optical_sar
50, unsupported 105, vqa 87. The validation split is **n = 86**. A 0.965116 validation accuracy over 86
examples, drawn from a 576-query corpus, is **indicative only**; it is not a benchmark result.

**Evidence:** `artifacts/router/router_adapter_v001/metadata.json` (`corpus`),
`artifacts/router/threshold_sweep_val.json`.

### L-68 — The change-VQA test and test2 splits share scenes (`MEASURED`)

The change-VQA dataset's integrity block lists `allowed_shared_pairs: [["Test", "Test2"]]` — the two
test splits legitimately share scenes, so they are **not independent draws** of the same population. A
number from one is not a confirmation of the other.

**Evidence:** `artifacts/change_vqa/run/run_record.json` (`dataset.integrity`).

### L-69 — The optical-SAR metric is on a 19-class space with five absent classes (`MEASURED`)

The fusion metric is defined over a **19-class** label space; the scored held-out split contains only 14
present classes (`classes_present: [0,2,3,4,5,6,7,8,9,10,12,13,17,18]`). The absent five contribute 0.0
to macro-F1 by construction, which is why accuracy (0.931) and macro-F1 (0.434161) diverge so sharply.
The metric is a single head's held-out result; it selects no head and compares no arms.

**Evidence:** `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`.

### L-70 — The grounding evaluation is a single dataset and a single expression family (`MEASURED`)

Grounding is measured only on VRSBench referring expressions (n = 16,159). Nothing measures grounding on
a different expression style or a different imagery distribution. The `matched6` protocol (top_k = 6) is
the only protocol variation recorded.

**Evidence:** `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`,
`eval_result_matched6.json`.

---

## 9. Explicit non-claims

These are things a reader might reasonably assume, which this project does **not** claim:

- **No claim of state-of-the-art performance** on any benchmark.
- **No claim of production readiness for model quality.** The deployment runs; the models carry the
  limitations above.
- **No claim that the trained heads generalise** beyond their training-family test splits.
- **No claim that calibration improves confidence** — it made ECE worse (0.013755 → 0.014929).
- **No claim that the VLM adapter is accepted** for production use — it is `ACCEPTANCE-REJECTED`.
- **No claim of an end-to-end accuracy number** — none exists.
- **No claim that the router is correct on all phrasings** — residuals exist (§2).
- **No claim that the router number is a test result** — it is validation, ungated, n = 86.
- **No claim that the optical-SAR or change-VQA rulings are settled** — both are `OPEN`.
- **No claim of robustness** to adversarial, corrupted, or out-of-distribution inputs.
- **No claim of geolocation accuracy** — grounding boxes are image-relative, not geodetic.
- **No claim that the system is a safety-, legal-, or life-critical tool.**
- **No claim that the code is licensed** — no `LICENSE` file exists.
- **No claim that the release repositories are public** — three of four are private by design.

---

## 10. Where the evidence lives

| Topic | Evidence |
|---|---|
| All measured metrics | `artifacts/**/*.json`, verified by `release/tools/verify_readme_metrics.py` |
| Metric honesty rules | [`BENCHMARKS.md`](BENCHMARKS.md), [`EVALUATION.md`](EVALUATION.md) |
| The grounding resolution rejection | `docs/PHASE7_RESOLUTION_DECISION.md` |
| The VLM rejection | `artifacts/vlm/phase6_closure.json` (`why_acceptance_rejected`), `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` |
| The optical-SAR ruling | `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json` |
| The change-VQA ruling | `artifacts/change_vqa/run/PROMOTION.json`, `docs/PHASE19_FINAL_HARDENING.md` §4.5 |
| The calibration result | `artifacts/calibration_v001.json`, `docs/STEP7_BACKEND_CHAIN_REPORT.md` §7 |
| B-07 / B-02 | [`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1, §8.2; `docs/FINAL_DELIVERY_TODO.md` §5 |
| Router residuals | [`architecture/04-router.md`](architecture/04-router.md), `LIVE_VALIDATION_POSTFIX.md` |
| Environment traps | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) §8 |
| Stale documentation | `release/CURRENT_RELEASE_STATE.md` §6, `docs/PHASE12_CURRENCY_CORRECTION.md` |
| The stale-claim class | `docs/PHASE12_CURRENCY_CORRECTION.md` §6 |
