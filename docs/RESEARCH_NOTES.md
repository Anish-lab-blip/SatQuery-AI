# Research Notes

Engineering findings, negative results, and design decisions that would otherwise be lost. Every one
was learned by **probe or execution**, not assumption, and every one is recorded so it is not
rediscovered. Where a finding changed code, the change is named; where a finding was *reported rather
than patched*, the reason is given.

**Status tags:** `MEASURED` · `RESOLVED` · `REJECTED` · `OPEN` · `ATTEMPTED`.

**How to read this document.** §2 lists the twelve numbered findings that changed the code
(F4-1…F5-3, P7-1, C-1, C-6…C-9), each with **what was measured**, **the consequence**, and **the file
that records it**. §3–§9 are the larger case studies: the pre-registered resolution rejection, the
router-defect fix, the harness false-positive, the `auto`-mode fallthrough, the
`interpret()`/`chooseTask()` asymmetry, the environment findings, and the BigEarthNet contradiction.

**Companions.** [`LIMITATIONS.md`](LIMITATIONS.md) (the exhaustive catalogue of what does not work),
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (the environment traps in operator form),
[`architecture/04-router.md`](architecture/04-router.md), [`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1.

---

## 1. The convention that matters most here

A finding is only recorded when it was **measured**, and the measurement is quoted rather than
summarised. Two examples of why this convention exists, both recorded in the source documents:

- A per-class guardrail (V2, below) was written with a 1.0 pp threshold that sits **below the
  measurement resolution** of its own statistic — it flagged three classes whose drops were
  statistically indistinguishable from zero. The rule was wrong, not the model.
- A "bounded, not fixed" classification of a path-disclosure finding was **wrong in three ways**, and
  the third mattered most: it was not bounded and not measured. The recorded lesson is in the wording:
  **`"bounded"` is a claim, and a claim is not a measurement.**

Where a value could not be established, the honest entry is *"not measured"* rather than a hedge.

---

## 2. Findings that changed the code

### F4-1 — the MiniLM tokenizer ceiling is 256, not 128 (`MEASURED`)

**What was measured.** The config said `router.max_length: 128`. A probe showed the MiniLM tokenizer's
**own ceiling is 256**. 128 is therefore a deliberate truncation *well inside* the ceiling, not the
model's limit — but the config read as if it were the latter.

**Consequence.** `router.max_length: 128` is retained, but `FrozenEncoder` now applies it explicitly
and **rejects** any value above 256, because truncating above the ceiling is a **silent no-op** — "a
control that appears to work and does nothing". Satellite queries are short, so halving the sequence
halves attention cost for no measurable accuracy loss.

**Recorded in.** `docs/PHASE4_ROUTER_REPORT.md` §F4-1; `docs/ARCHITECTURE_FREEZE.md`;
`artifacts/router/router_adapter_v001/metadata.json` (`encoder.max_length: 128`).

### F4-2 — the router does not need a GPU (`MEASURED`)

**What was measured.** The encoder is frozen, so embeddings are a pure function of the query text.
Embedding the corpus measured **0.118 s / 64 queries on CPU**; training the **51,725-parameter** adapter
on the cached vectors measured **0.28 s for 20 epochs over 4,096 vectors**.

**Consequence.** The router can be retrained during development at **zero GPU quota cost**. The plan's
budget line — "Router | CPU/T4 | <1 h" — was roughly **three orders of magnitude pessimistic**; Phase 4
ran to completion locally.

> **Number discipline.** `artifacts/router/router_adapter_v001/metadata.json` records
> `num_parameters: 51725`. The measured figure is **51,725**; use it rather than any other recollection
> of the parameter count.

**Recorded in.** `docs/PHASE4_ROUTER_REPORT.md` §F4-2;
`artifacts/router/router_adapter_v001/metadata.json`.

### F4-3 — the corpus needs *group*-level splitting (`MEASURED`)

**What was measured.** Template-generated queries are near-duplicates. Splitting by example would put
`"Show me the water body."` in train and `"Show me the road."` in val — one token apart — and report a
**fake** accuracy.

**Consequence.** Splits are by **group** (template id, or hard-negative family), never by example,
mirroring `evaluation.leakage.assign_splits_by_scene` deliberately: "the failure mode is identical, so
the guard should look identical". Hard-negative families are placed in the **test** split so their
accuracy measures generalisation rather than memorisation. The corpus metadata records
`groups: 54` over `total: 576` queries.

**Recorded in.** `docs/PHASE4_ROUTER_REPORT.md` §F4-3;
`artifacts/router/router_adapter_v001/metadata.json` (`corpus.groups`).

### F5-1 — `AutoModelForVision2Seq` is absent, not deprecated (`MEASURED`)

**What was measured.** In transformers **5.17.0**:

| Class | Present? |
|---|---|
| `AutoModelForImageTextToText` | **present** |
| `AutoModelForVision2Seq` | **ABSENT** |
| `AutoModelForMultimodalLM` | present |

The class **does not exist** (it is not merely deprecated).

**Consequence.** The loader is resolved by **feature detection**, never hardcoded to one class name, so
a transformers version that renames or removes a class does not break the load path.

**Recorded in.** `docs/PHASE5_VLM_CONTRACT.md` §F5-1.

### F5-2 — the processor cost overrun is ~17×, not 4× (`MEASURED`)

**What was measured.** The processor's default `longest_edge` is **2048**, which upscales a 512-px tile
**4×** and then splits it (`do_image_splitting=True`) into sub-images:

```
INPUT: one 512×512 RGB tile

DEFAULT  (size.longest_edge = 2048, do_image_splitting = True)
    pixel_values  (1, 17, 3, 512, 512)     <- 17 images
    prompt tokens 1142

PINNED   (processor_longest_edge = 512)
    pixel_values  (1, 1, 3, 512, 512)      <- 1 image
```

The plan estimated a 4× cost overrun; the real figure is **~17×**.

**Consequence.** `processor_longest_edge` **must** be set explicitly on the processor at construction
time, and `core/config.py` now **enforces** `processor_longest_edge <= image.tile_size` so the pin is a
control rather than a comment. Confirmed in the real load path: `max_images_seen = 1` — "the F5-2 pin
holds in the real load path, not just the probe. Unpinned it would read 17."

**Recorded in.** `docs/PHASE5_VLM_CONTRACT.md` §F5-2 (and the `max_images_seen` confirmation);
`docs/ARCHITECTURE_FREEZE.md` (`vlm.processor_longest_edge: 512`).

### F5-3 — SmolVLM requires `<image>` tokens in the prompt (`MEASURED`)

**What was measured.** Hand-written prompt strings fail:

```
ValueError: The total number of <image> tokens in the prompts should be the
same as the number of images passed. Found [0] <image> tokens and [1] images
```

**Consequence.** Prompts are **always** built through `processor.apply_chat_template()`, enforced by
the config key `vlm.prompt_must_use_chat_template: true`.

**Recorded in.** `docs/PHASE5_VLM_CONTRACT.md` §F5-3.

### P7-1 — the RemoteCLIP projected dimension is 512, not 768 (`MEASURED`)

**What was measured.** `visual.positional_embedding` is **768 wide**, but `visual.proj` is **(768, 512)**
— the embeddings comparable against the text tower are the **projected** ones, i.e. **512**
(`dim_match: True`). The grounding head's per-cell feature is therefore `4 × 512 = 2048`.

**Consequence.** Using 768 anywhere in the grounding path would be a **silent shape error**, caught only
at the similarity computation — after the patch features have already been computed and cached.
`grounding.encoder_projected_dim: 512` is declared in config so `core/config.py` can validate the head
**without importing torch**, and `RemoteCLIPEncoder._verify_contract()` asserts the same value against
the real model at load time.

**Recorded in.** `docs/PHASE7_GROUNDING_CONTRACT.md` §P7-1; `docs/ARCHITECTURE_FREEZE.md`.

### C-1 — the availability mask is consumed by the head, not by CROMA (`MEASURED`)

**What was measured.** CROMA always sees the canonical channel counts (**12 optical, 2 SAR**,
zero-filled to canonical order). The availability mask is applied by the **fusion head**, not the
encoder. The verified fusion-head input dimensions:

```
optical_GAP      (B, 768)
SAR_GAP          (B, 768)
joint_GAP        (B, 768)
optical_mask     (B, 12)     <- availability, from the sensor adapter
sar_mask         (B,  2)     <- availability, from the sensor adapter
                 ---------
concat           (B, 2318)
```

i.e. `input_dim = 3 × 768 + 12 + 2 = 2318`.

**Consequence.** Channel/band dropout during fusion-head training is **mandatory** — "it is what teaches
the head to trust the availability mask". The sensor adapter never fabricates a missing band; missing
channels are masked/zero-filled per the validated adapter policy.

**Recorded in.** `docs/ARCHITECTURE_FREEZE.md` §2.5 (C-1).

### C-6 — T4 is SM 7.5, so training uses fp16, not bf16 (`MEASURED`)

**What was measured.** The target GPU (Tesla T4) is compute capability **7.5**; bf16 tensor cores are
**unavailable** there.

**Consequence.** `training.precision: fp16`. The loader validates the value is one of `fp16|bf16|fp32`.
The fp16-vs-planned-bf16 deviation is documented as a finding in its own right, not hidden.

**Recorded in.** `docs/ARCHITECTURE_FREEZE.md` (`training.precision` row, C-6);
`docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §6.3.

### C-7 — `image_resolution % 8 == 0` (`MEASURED`)

**What was measured.** CROMA requires `image_resolution % 8 == 0`. The native value **120** satisfies it
and yields **225 patches**.

**Consequence.** Enforced at config load (`croma.image_resolution: 120`).

**Recorded in.** `docs/ARCHITECTURE_FREEZE.md` (`croma.image_resolution` row, C-7).

### C-8 — ZeroGPU does not support `torch.compile` (`MEASURED`)

**What was measured.** `torch.compile` must never be enabled on the (historical) ZeroGPU target.

**Consequence.** Enforced: the loader **fails startup** if `deployment.torch_compile` is true. This is
why setting `SATQUERY_TORCH_COMPILE=true` fails startup rather than silently taking effect (see
[`DEPLOYMENT.md`](DEPLOYMENT.md) §6.3).

**Recorded in.** `docs/ARCHITECTURE_FREEZE.md` (`deployment.torch_compile` row, C-8);
`configs/deploy.yaml` header.

### C-9 — STANet hyperparameters are upstream-verified (`MEASURED`)

**What was measured.** Change detection uses a STANet-style architecture with **upstream-verified**
hyperparameters: ResNet-18 encoder, **PAM** self-attention mode, tile **256** (non-overlapping),
threshold **0.50**, loss `0.5·BCE + 0.5·Dice`, `lr = 1e-3`, `batch_size = 8`. The LEVIR-CD split is
**7120 / 1024 / 2048** (256-px patches). Post-processing: threshold → morphological cleanup → connected
components → minimum-component filter.

**Consequence.** The implementation is **reimplemented, not vendored** — so the upstream hyperparameters
are pinned as the contract, and the trained head's embedded metadata carries the same values
(`sa_mode: "PAM"`, `width: 128`, `encoder: "resnet18"`).

**Recorded in.** `docs/ARCHITECTURE_FREEZE.md` §2.5 (C-9);
`artifacts/change/eval_test/eval_result.json` (`checkpoint_embedded_config`).

---

## 3. The grounding resolution decision — a pre-registered rejection (`REJECTED`)

**Question.** Should grounding decode at **448** or **224**?

**Answer: 224. 448 was rejected** — notable because the rejection was *pre-registered* and then
*confirmed* by a paired test over identical samples (n = 16,159), on a Tesla T4
(`--all --device cuda --tag full`).

### 3.1 The rule, fixed before the result was seen

```
448 WINS  if Recall@0.5 improves by >= 0.05 absolute
          OR mean best IoU improves by >= 0.05 absolute
224 WINS  otherwise
INCONCLUSIVE if fewer than 30 samples were scored
```

The artifact records `rule_changed_since_preregistration: false` — the rule was **not** modified after
the result was seen.

### 3.2 The result

```
224 WINS
  recall@0.5 gain 448/224 : -0.0022
  bestIoU    gain 448/224 : -0.0147
  latency          ratio  : 1.59x
```

Neither component came close to the +0.05 margin. Both were **negative**.

### 3.3 Measured detail

| metric | 224 | 448 | delta |
|---|---|---|---|
| token grid | 7 × 7 = 49 | 14 × 14 = 196 | 4.0× tokens |
| attention cost (n²) | 1× | 16× | — |
| with boxes | 16159/16159 | 16159/16159 | — |
| **mean best IoU** | **0.0972** | 0.0825 | **−0.0147** |
| Recall@0.10 | **0.3298** | 0.2599 | **−0.0699** |
| Recall@0.25 | **0.1187** | 0.0944 | **−0.0243** |
| Recall@0.50 | **0.0234** | 0.0212 | **−0.0022** |
| matched IoU | 0.0972 | 0.0825 | −0.0147 |
| latency mean | **20.0 ms** | 31.8 ms | 1.59× |
| latency p90 | **20.9 ms** | 32.9 ms | 1.57× |
| peak VRAM | **592.1 MB** | 599.8 MB | +7.7 MB |
| wall time | **~8.5 min** | ~11.2 min | 1.32× |

**448 is worse on every quality metric and slower. There is no axis on which it wins.**

Best-IoU distribution — the shift is a whole-distribution move toward the zero-overlap bucket, not a
tail effect:

| bucket | 224 | 448 |
|---|---|---|
| 0.00–0.10 | 10,829 | 11,957 |
| 0.10–0.25 | 3,412 | 2,677 |
| 0.25–0.50 | 1,540 | 1,182 |
| 0.50–0.75 | 336 | 307 |
| 0.75–1.01 | 42 | 36 |

The 224 column dominates the top three buckets; 448 has ~1,100 more near-total misses.

### 3.4 Paired analysis — independent confirmation

Both resolutions scored the **same 16,159 samples**, so the paired test removes between-object variance:

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
escalate: both say 224, and the confidence interval excludes zero by a wide margin. The win/loss split
is also informative: 448 wins on only **8.5 %** of records and loses on **20.9 %** — the finer grid is
not merely neutral, it is **actively harmful on a fifth of the corpus**.

### 3.5 Recall ladder, paired

| threshold | 224 | 448 | diff | 95% CI |
|---|---|---|---|---|
| 0.10 | 0.3298 | 0.2599 | −0.0699 | excludes zero |
| 0.25 | 0.1187 | 0.0944 | −0.0243 | excludes zero |
| 0.50 | 0.0234 | 0.0212 | −0.0022 | excludes zero |

The gap **narrows as the threshold rises** — the signature of a method that cannot reach high IoU either
way. At IoU 0.50 the two are within 0.002 of each other and both are near the floor.

### 3.6 Why 448 did not help — the honest reading

The zero-shot method selects a patch by text similarity and returns that patch's box. At 224 a box is
1/7 of the image; at 448 it is 1/14. Two things work against the finer grid:

1. **The peak is not sharper at 448.** Splitting each cell into four gives four chances to pick a wrong
   sub-cell, and the similarity field on frozen features is smooth, so the argmax moves around. 448 wins
   on 8.5 % and loses on 20.9 % — losses outnumber wins by 2.5 : 1.
2. **Recall@0.10 drops the most (−0.0699).** If finer tokens genuinely localised better, the *loosest*
   threshold would benefit most. It degrades most, which means the fine grid adds positional noise
   rather than positional precision.

This is the **zero-shot baseline's** limitation, not a property of RemoteCLIP. A **learned** head
trained to regress boxes from these features may respond differently.

### 3.7 What this establishes, and what it does not

Establishes: grounding runs at **224** (frozen in `configs/base.yaml`:
`grounding.image_size: 224`, `grounding.resolution_frozen: true`); peak VRAM for the frozen encoder at
224 is **592 MB**; encoder latency at 224 on a T4 is **20 ms/image** (5× faster than the CPU figure of
97 ms); the 224 localisation floor is **1/7 of image width per token**.

Does **not** establish: whether the zero-shot baseline is *good* (it is not — mean best IoU 0.0972 and
Recall@0.5 0.0234 are weak, and this is an ablation floor for the Phase-8 head, not a product); whether
a **trained** head has the same resolution sensitivity (re-opening the question after Phase 8 is
legitimate *if* the head's validation curve suggests it, and would be a **new** pre-registered
experiment, not a silent retune); anything about hidden ISRO/SAC imagery (VRSBench is overhead optical;
the hidden set is Cartosat-2S + RISAT, a different distribution entirely).

**Degeneracy note.** At n=12, n=40 and n=6 the smoke runs reported `Recall@0.5 = 0.0000` at **both**
resolutions and the script emitted a degeneracy warning; at full scale the metric is non-zero
(0.0234 / 0.0212), so the note correctly did not fire. The sub-floor runs were never treated as
evidence.

**Reproduction.**

```bash
python scripts/exp_grounding_resolution.py \
    --vrsbench  <data-root> \
    --checkpoint <RemoteCLIP-ViT-B-32.pt> \
    --all --device cuda --tag full

python scripts/analyze_grounding_resolution.py --tag _full
```

Artifacts: `per_sample_224_full.jsonl`, `per_sample_448_full.jsonl`,
`resolution_experiment_full.json` — 16,159 lines each. Every aggregate is recomputable from the JSONL
without re-running the encoder.

**Recorded in.** `docs/PHASE7_RESOLUTION_DECISION.md` (full document).

---

## 4. The router defect — a real bug, found and fixed (`RESOLVED`)

### 4.1 Symptom

The query *"Where are the built-up areas in this image?"* — with **one** asset attached — collapsed to
**`vqa`** and answered **"River"**, instead of routing to `grounding`. A second query, *"Where is the
new airport?"*, behaved the same way.

### 4.2 Root cause

Two functions with different information:

- **`interpret()`** — produces the console's *reading*; **asset-count-blind** (text only).
- **`chooseTask()`** — performs *dispatch*; **asset-count-aware**.

The defect was in the reading/dispatch path's handling of spatial/lexical cues. The pre-fix replay of
the shipped functions shows the mechanism exactly:

```
"Where are the built-up areas in this image?" — one asset.
Before: reading=change, temporal=required → dispatched=vqa (wanted=change_vqa, substituted=true).
        \bbuilt\b matched the temporal regex and `area` matched inside "areas".
```

So the word "built" was in the temporal (change) set, and "area" matched as a substring of "areas".
Region queries fell through to the generic VQA specialist.

### 4.3 Fix and verification

The fix was deployed to `SatQuery-Frontend` (`mission.js` `interpret()`, `ff46eba42b18`) and the sibling
`core.js` `SQ.policy` (`2d7ae53b482d`). It was validated by **three independent live passes**:

| Pass | Deployed HEAD | Result |
|---|---|---|
| 1 | `ff46eba42b18` + `d413d3672311` | 8/8 |
| 2 | `2d7ae53b482d` | 8/8 |
| 3 | `2d7ae53b482d` | 8/8 |

Both defect queries now dispatch to `grounding`:

| Query | Pass-1 run id | Pass-2 run id | Pass-3 run id |
|---|---|---|---|
| Where are the built-up areas in this image? | `run_f0d7a90b5aa1` | `run_2a07dcdbae96` | **`run_467ffa406f22`** |
| Where is the new airport? | `run_69e38a182a71` | `run_9134f40a258c` | **`run_46980ba55c62`** |

**24 live runs, 24 correct dispatches, 0 mock nodes.** No run id is shared between passes. Post-fix the
live answers read `reading=grounding, temporal=none` and `[grounding] Located 6 candidate region(s) …
Highest objectness 0.82` / `0.83`. Screenshots (`A1…B2*.png`) are in the live-validation scratch
directory.

### 4.4 The residuals the fix left (and did not hide)

- *"What is the new runway?"* still reads `change` rather than `vqa` (the `new`-as-change heuristic
  fires on non-`where` questions). Strictly better than pre-fix, where `new` was unconditionally
  temporal. "A lexical router cannot cleanly separate 'the new X' from 'what's new'."
- *"How much built-up area was added?"* now reads `vqa` (under-trigger), because `built` was dropped
  from the temporal set and `area` no longer matches inside `areas`.

**Recorded in.** `docs/FINAL_DELIVERY_TODO.md` §5 B-08, §6 E-09/E-11/E-14;
`.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md`; `run_final2.txt`, `run_final3.txt`;
`results_final.json`, `results_pass3.json`.

### 4.5 The corpus the fix was validated against (and its limits)

The router is trained and validated on a **576-query** corpus in **54** groups
(`corpus_total: 576`, `corpus_groups: 54`), distributed by task: caption 91, change 115, grounding 128,
optical_sar 50, unsupported 105, vqa 87. The validation split used for the reported accuracy is
**n = 86**; the corpus is flagged `corpus_limited: true`. The adapter has **51,725** parameters over a
384-dim frozen MiniLM encoder (6 task classes, 4 modalities, 3 binary heads).

This is why the router number must be read as **indicative only** — see
[`LIMITATIONS.md`](LIMITATIONS.md) §1 (L-12) and §8 (L-67). The fix was validated by **live behaviour**
(24 runs, 24 correct dispatches), not by a corpus accuracy jump, precisely because the corpus is small.

**Recorded in.** `artifacts/router/router_adapter_v001/metadata.json`;
`artifacts/router/threshold_sweep_val.json`.

---

## 5. The harness false-positive — caught before it could lie (`RESOLVED`)

### 5.1 What happened

An earlier live-validation harness typed queries with **synthetic CDP key events**, which Chrome
**silently drops when the window lacks OS focus**. The harness therefore dispatched the page's *default*
query and still recorded a "result" — a **false pass**. Measured directly: with Chrome backgrounded,
`press_key("Z")` left `#qtext.value` unchanged, while `type_text("Q")` (CDP `Input.insertText`, not
focus-gated) inserted fine. The re-run attempt failed on case 1 with `run_id=0002`, `mock_nodes=9`,
`answer="No answer yet"`, and only a `capabilities` call — the **mock path**.

### 5.2 Fix

The current harness **asserts form state before dispatch** (`q_ok`, `obs_ok`, `t0_ok`) and uses
deterministic query entry (`js()` value-set + `type_text()` via CDP `Input.insertText`). Two further
harness bugs were found and fixed, both causing **false failures**:

- the answer `[task]` tag exists only for region tasks (vqa/caption answers are bare); and
- the intent panel is a **concatenated** string, so `task([a-z_]+)` must be matched **non-greedily** up
  to `modality`.

The harness now computes the dispatched task as `answer_tag` when present, else the reading.

### 5.3 Independent check

The earlier 8/8 run was re-examined and confirmed **not** infected — its intents were query-specific
(e.g. A1 read `taskvqa…temporalnone`, not the default's `taskchange…temporalrequired`), its answers
embedded the query text, and A6's answer proved two files were uploaded. The failure mode is recorded
because it is exactly the **silent false-positive an evaluation harness must never have**.

**Recorded in.** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Why this pass
needed a new harness"); `docs/FINAL_DELIVERY_TODO.md` §6 E-14.

---

## 6. The `transport_mode: auto` fallthrough (`OPEN`)

`SATQUERY_TRANSPORT=auto` tries the tunnel, then falls through to the forward path on timeout. The
forward path to a **private** repo returns `302` quickly, but the wake step still consumes
`SATQUERY_WAKE_TIMEOUT_S` (120 s) first — so a worst-case failed request takes ≈ **249 s**
(150 + 120). This is the **root shape** of the observed transient tunnel gap.

The recorded root cause is precise: in `auto` transport mode a tunnel timeout **falls through** to the
forward path (`SatQuery-Backend/main.py:546`), which then burns `wake_timeout_s=120` on a `302` → the
observed `504`.

A patch (`fix-b07-forward-unavailable.patch`) was authored and verified (`git apply --check` clean,
`py_compile` clean, applies to the deployed `89d80eaddec5`). It adds `forward_unavailable` (**503**,
terminal `302`/`401`/`403` on the forward path) and `upstream_timeout` (**504**, tunnel healthy but
slow) codes, plus the `codespace_name` `.strip()` fix.

> **Status: `OPEN`.** The patch is **prepared but NOT deployed.** The deployed health payload still
> shows the trailing `\n`.

**Recorded in.** `release/CURRENT_RELEASE_STATE.md` §6; `docs/FINAL_DELIVERY_TODO.md` §5 B-07;
[`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1.

---

## 7. The `interpret()` / `chooseTask()` asymmetry — intentional (`RESOLVED`)

For *"What changed between the earlier and later image?"* with **one** asset attached, the console
**reads** `change` while dispatch correctly falls back to **`change_vqa`**. This is **not** a bug:

- the **reading** describes the question's intent (asset-count-blind);
- the **dispatch** respects what can actually be computed with the assets present (asset-count-aware).

It is documented so it is not mistaken for a defect. In the live-validation table this appears as the
one case where reading ≠ dispatched (`change` → `change_vqa`) — "flagged, not failed".

**Recorded in.** `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` ("Discriminator
note"); [`architecture/04-router.md`](architecture/04-router.md).

---

## 8. Environment findings (would otherwise cost hours)

| Finding | Detail |
|---|---|
| **Dead proxy in the authoring sandbox** | outbound calls need `--noproxy '*'` (curl) or `ProxyHandler({})` (Python). |
| **The sandbox proxy is slow for uploads** | `huggingface_hub` uploads stalled at ~51 kB/s through `http_proxy=127.0.0.1:58294`; the fix is to unset `http_proxy`/`https_proxy` and set `no_proxy='*'`. |
| **`hf_hub_download` returned an EMPTY file** | sha256 `e3b0c442…` (the empty-content hash), which produced a **false FAIL** for all six artifacts. The honest check is a direct HTTPS download with `ProxyHandler({})`. |
| **pytest is only in the repo venv** | `.venv/Scripts/python.exe`; a bare `pytest` misses it. |
| **The full test suite trips a bulk-delete guard** | sandbox-specific; affects `test_safe_delete_shim`. |
| **Cloudflare `_headers` concatenate** | two matching rules are merged, not overridden; Chromium takes the **first** `max-age`. |
| **Cloudflare 308-redirects `X.html` → `/X`** | reference the extensionless path. |
| **A forwarded Codespace port returns `302`** | for a private repo — this is *why* the tunnel exists. |
| **Chrome drops synthetic CDP key events without OS focus** | the harness false-positive (§5). |
| **`browser-use` block-buffers stdout** | even when redirected; needs explicit line buffering to stream. |
| **WDAC blocked `orjson`, so FastAPI could not import** *(historical — no longer reproduces)* | `ImportError: DLL load failed while importing orjson: An Application Control policy has blocked this file.` Verified as `OSError [WinError 4551]` from a direct `ctypes.CDLL` on the binary. FastAPI's own guard catches `ModuleNotFoundError`, not a policy-blocked `ImportError`, so it aborted the import. When the block lifted, **four live defects became reachable** that had been sitting in `gateway/` the whole time (G-1…G-4). Do not cite this as a current limitation. |
| **`pandas._libs.parsers` / `sparsefuncs_fast` App Control block** | 56 evaluation-side tests remain uncollectable. Unchanged; unrelated to the backend chain. |
| **CPU-only torch** | CUDA autocast is a no-op; no GPU path was exercised in the authoring environment. |

### 8.1 A withdrawn explanation, recorded rather than deleted

The reason that nine backend-chain tests (D–I, Q, R) were NOT RUN was originally recorded as "the
environment's egress proxy intercepts outbound HTTP and returns `502`". That explanation was
**re-measured on 2026-09-22 and does not hold**: `https://huggingface.co` returned **HTTP 200**,
`/api/models` returned 200 with real JSON, and a real weight-file path returned 200. The correct reason
is that **no upstream exists** — no Space and no service has ever been deployed, and the gateway is a
pure proxy that holds no second copy of the capability table, so a metadata request *is* an upstream
request. The classification (NOT RUN / ENVIRONMENT-BLOCKED) is unchanged; only the reason changed.

### 8.2 The transferable lesson from the stale-negative class

Three documents written after the Phase-12 A/B experiment still described it as un-run, even though it
had completed 34 hours earlier. **A stale negative claim is more dangerous than a stale positive one**:
a wrong number is eventually contradicted by the artifact it describes, but "this has never been
executed" is contradicted by *nothing* — no test fails, no hash moves, no invariant breaks. The rule
this suggests: **never assert that an artifact does not exist from memory; assert it from a command, and
record the command.** No test can check that a documented absence is still absent.

**Recorded in.** `docs/PHASE19_FINAL_HARDENING.md` §5 (historical blocker) and §5.3 (CPU-only torch);
`docs/STEP7_BACKEND_CHAIN_REPORT.md` §5, §11, §13, §15; `docs/PHASE12_CURRENCY_CORRECTION.md` §6;
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 9. The BigEarthNet format contradiction (`ATTEMPTED`, reported not resolved)

The BigEarthNet data format **contradicts the original plan**. This was **reported rather than silently
patched**, because quietly changing the preprocessing would move the frozen config hash. Two facts
matter:

1. **The percentile stretch is not upstream.** The BigEarthNet documentation — its uses, mentions, or
   endorsements — does **not** specify a percentile stretch. This project nevertheless applies
   percentile normalisation (**2/98**) for optical inputs to match the CROMA contract. That is a
   **deliberate, documented choice**, not an upstream fact.
2. **The local subset is single-label.** It is **100 % single-label** against the official **1–11
   multi-label** scheme, so metrics computed on it are **not comparable** to published multi-label
   numbers.

**Consequence.** The contradiction is recorded as a limitation and an open item rather than resolved by
editing preprocessing (which would move `Config.hash` off `78f1e3700da15aa1`).

**Recorded in.** [`LIMITATIONS.md`](LIMITATIONS.md) §3 (L-24);
`docs/PHASE12_LABEL_POLICY_DECISION.md`; `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md`.

---

## 10. Additional measured defects that shaped the design

These are not headline findings, but each changed a decision or a guard. They are recorded together
because they share a shape: a value or a claim that was produced without a measurement, then corrected
by one.

### 10.1 The train/serve skew (the F2 finding) (`RESOLVED`)

**What was measured.** `scripts/prepare_change_vqa.py` builds its change features from the **trained**
STANet (`DEFAULT_CHANGE_CHECKPOINT`), but serving had no equivalent wiring: `change.checkpoint_path` is
unset in `configs/base.yaml`, so the registry passed no `checkpoint_path` and the change-VQA specialist
would construct an **untrained** STANet and answer from a representation the head was never fitted on.

**Consequence.** `app/serving.py::_wired_change_vqa_builder` supplies the **same** checkpoint the
change builder uses, so the detector behind a change answer and the detector behind the head's training
features are one artifact by construction. The specialist's own `feature_spec_mismatch()` check stays
armed as a second line of defence, not the only one.

**Recorded in.** `app/serving.py` (`_wired_change_vqa_builder` docstring).

### 10.2 The optical-SAR encoder was unreachable by default (`RESOLVED`)

**What was measured.** `specialists/optical_sar/specialist.py` builds CROMA only when handed a
`checkpoint_path` that exists. `croma.checkpoint_path` is not in `configs/base.yaml` — and must not be,
or `Config.hash` moves — so the registry passed no path, the gate was False, and the default serving
composition ran with `encoder=None` while the checkpoint sat on disk the whole time. The registry then
correctly reported `DEGRADED` ("no encoder; running on fallback"): a deployment that could never answer
an optical/SAR question.

**Consequence.** `app/serving.py::_wired_optical_sar_builder` resolves the checkpoint from the **pinned
identity** through the hash-exempt channel (`croma.resolve_checkpoint_path`: env → config → pinned Hub
cache, offline first) and wires the fusion head the same way. The resolution `source` is logged so
"which checkpoint did this process actually use" is answerable from the trace.

**Recorded in.** `app/serving.py` (`_wired_optical_sar_builder` docstring);
`specialists/optical_sar/specialist.py`.

### 10.3 G-1 … G-4 — four defects the first real ASGI run found (`RESOLVED`)

**What was measured.** The ASGI layer had **never executed** in the repository, and the reason was
recorded in the code as an environment blocker (a WDAC block on `orjson.pyd`). When the block lifted,
within the first hour of actually running the server **four real defects surfaced**, four of them latent
for as long as the blocker was believed:

| ID | Defect | Severity |
|---|---|---|
| **G-1** | `request: Request` never resolved — an in-function import left `Request` out of `__globals__`, so FastAPI silently reinterpreted the parameter as a required **query** parameter named `request`; **every POST body was misread** and no handler ran | **Critical** |
| **G-2** | an unsupported `force_task` enum value was forwarded upstream instead of refused locally → wrong status, wasted round trip | High |
| **G-3** | an empty `HF_TOKEN` produced `Authorization: Bearer `, which httpx rejects → a crash reported as an upstream failure | High |
| **G-4** | a non-JSON upstream error body was relayed verbatim → contract break and internal-text disclosure | High |

**Consequence.** All four fixed, with regression tests that assert the **cause** rather than the
symptom. The transferable lesson: "a documented blocker is a place where evidence stops, and nothing was
watching for the blocker to lift."

**Recorded in.** `docs/STEP7_BACKEND_CHAIN_REPORT.md` §13, §14.

### 10.4 F-1 … F-7 — the gateway layer-crossing audit (`RESOLVED` / recorded)

**What was measured.** A series of findings, each found by driving the real ASGI stack rather than by
reading:

| ID | Finding | Disposition |
|---|---|---|
| **F-1** | a vacuous assertion in the gateway's own test suite | fixed |
| **F-2** | an upstream CORS header bypassed the allowlist entirely (**security**) | fixed |
| **F-3** | `404`/`405` did not carry the contract's error envelope | fixed (gateway); `app/space_app.py` does not register the handler (bounded defect F-12b) |
| **F-4** | `rate_limited` was emitted but documented nowhere | fixed |
| **F-5** | the per-IP rate limiter is defeated by a client-supplied `X-Forwarded-For` (measured 5/8 throttled without the header, **0/8** with a fresh value per request) | **ruled**: fairness only, not a security control |
| **F-6** | the body-size cap was **declarative**, not enforced — an omitted `Content-Length` buffered the entire body past the cap (measured: 12 MiB body → 13.9 MiB peak; allocation tracked body size with no ceiling) | fixed with a streaming check alongside the header check |
| **F-7** | two parsers of one variable (`SATQUERY_MAX_FILE_BYTES`) diverged in opposite directions | fixed; both layers now refuse a malformed value and name the variable |

**Consequence.** The rate limiter's claim was narrowed to what the measurement supports, and the size
cap is now enforced twice — deliberately, because "the header check protects the gateway's memory
against honest clients, not against hostile ones".

**Recorded in.** `docs/STEP8_FINAL_CONFORMANCE_AUDIT.md` §15–§17;
`docs/DEPLOYMENT_ARCHITECTURE.md` §5.2, §4 (F-6 note).

### 10.5 F-13 … F-17 — what an unauthenticated client could learn (`RESOLVED` / recorded)

**What was measured.** A pass-by-pass audit of every value written to a client-visible field found that
`trace.inputs` (F-13), `trace.steps[PARSE].detail["inputs"]` (F-14), and the construction-failure
exception string (F-15) published **server-side filesystem paths** to an unauthenticated client —
F-15's case was "strictly worse" because the path was purely server-side and reachable on the **first**
request. F-16 found `result.change_map` / `evidence[].artifact_ref` carrying a real filesystem path
where the contract promised an `artifact://` URI. F-17 found `change_vqa.artifact_dir` configured but
never read.

**Consequence.** The owner ruled: sanitize client-facing exception messages to a **basename** (not a
replacement — the reason survives) and log the raw detail server-side; set an unavailable `artifact_ref`
to `null` with an explicit non-retrievable warning; do **not** fabricate `artifact://` URIs. The F-15
ruling needed **four** carriers, not three, because `result.execution_trace = trace` serializes the
trace object **twice**. F-17 is **documented, not patched**.

**Recorded in.** `docs/DEPLOYMENT_ARCHITECTURE.md` §5.3–§5.6; [`SECURITY.md`](SECURITY.md).

### 10.6 The single capability authority (`RESOLVED`)

**What was measured.** Two independent producers answered "what can this deployment do?":
`AnalysisController.health()` reported **6** capabilities from the registry; `space_app.describe_deployment()`
reported **2** from two `Path.exists()` calls — and only the latter was served. Also measured:
`HealthStatus(**controller.health())` **fails** with 3 `extra_forbidden` errors.

**Consequence.** Owner ruling: the **registry is authoritative**, and `app/deployment.py` is the single
adapter that derives contract vocabulary from the registry's spec table and the filesystem. This is the
"three-copy problem": any table that exists in two places will drift, and this one already had.

**Recorded in.** `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3.1;
`docs/STEP7_BACKEND_CHAIN_REPORT.md` §16.

### 10.7 T-1 / T-2 — two test-side findings worth carrying (`RESOLVED`)

**What was measured.** (T-1) An earlier note recorded "22 documented error codes"; the real taxonomy and
the contract both have **23**. The test now parses the contract's §5.2 table and compares it against
`core/errors.py`, so the two must agree. (T-2) Three documentation tests initially failed by matching the
**prohibition itself** (e.g. a search for `gr.Blocks` matched the docstring that forbids it); the tests
now parse the module with `ast` and strip docstrings before searching — "a search for a forbidden token
must run over executable code".

**Consequence.** Both corrections became **bidirectional** checks rather than one-off fixes.

**Recorded in.** `docs/STEP7_BACKEND_CHAIN_REPORT.md` §3, §13; `docs/PHASE19_FINAL_HARDENING.md` §3.7.

### 10.8 The calibration measurement, stated exactly (`MEASURED`)

**What was measured.** Temperature scaling was fitted on the Val split: `temperature = 0.9772731820958189`,
`n_samples = 16441`, `ece_before = 0.013755`, `ece_after = 0.014929`,
`ece_improvement = −0.001174` (**worse**), `nll_before = 0.6897411`, `nll_after = 0.6896308`,
`nll_improvement = +0.0001104`, `hit_bound = false`, `effective = true`. The raw softmax was **already
near-calibrated**, and temperature scaling made ECE very slightly worse while improving NLL marginally.

**Consequence.** The path is live (`core/controller.py` → `EvidenceEngine`), so a deployed result
carries a calibrated value; but this is a **measurement, not a quality judgment**, and must not be
described as scaling being "more accurate".

**Recorded in.** `artifacts/calibration_v001.json`; `docs/STEP7_BACKEND_CHAIN_REPORT.md` §7.

---

## 11. Where the evidence lives

| Topic | Evidence |
|---|---|
| Findings F4-1…F4-3 | `docs/PHASE4_ROUTER_REPORT.md` |
| Findings F5-1…F5-3 | `docs/PHASE5_VLM_CONTRACT.md` |
| Finding P7-1 | `docs/PHASE7_GROUNDING_CONTRACT.md` |
| Findings C-1, C-6…C-9 | `docs/ARCHITECTURE_FREEZE.md` |
| CROMA normalisation upstream evidence | `docs/CROMA_NORMALISATION_UPSTREAM_EVIDENCE.md` |
| The 448-vs-224 paired test | `docs/PHASE7_RESOLUTION_DECISION.md` |
| The VLM rejection | `docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md`, `artifacts/vlm/phase6_closure.json` |
| Router defect + 3 live passes | `.workbuddy-ai/scratch/live_validation/` (`run_output.txt`, `run_final2.txt`, `run_final3.txt`) |
| The undeployed B-07 patch | session scratch `fix-b07-forward-unavailable.patch` |
| Live validation harness | `.workbuddy-ai/scratch/run_all_postfix2.harness`, `recompute_verdicts.py` |
| The withdrawn egress explanation | `docs/STEP7_BACKEND_CHAIN_REPORT.md` §5 |
| The stale-negative lesson | `docs/PHASE12_CURRENCY_CORRECTION.md` §6 |
| The BigEarthNet contradiction | `docs/PHASE12_LABEL_POLICY_DECISION.md`, `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` |
| The gateway layer-crossing findings (F-1…F-7) | `docs/STEP8_FINAL_CONFORMANCE_AUDIT.md` §15–§17 |
| The ASGI defects (G-1…G-4) | `docs/STEP7_BACKEND_CHAIN_REPORT.md` §13 |
| The path-disclosure rulings (F-13…F-17) | `docs/DEPLOYMENT_ARCHITECTURE.md` §5.3–§5.6 |
| The train/serve skew and optical-SAR wiring | `app/serving.py` |
| The calibration measurement | `artifacts/calibration_v001.json`, `docs/STEP7_BACKEND_CHAIN_REPORT.md` §7 |
