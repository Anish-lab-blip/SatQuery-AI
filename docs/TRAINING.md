# Training — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED`
· `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

SatQuery AI trains **six** artifacts. Four are task heads, one is an intent-router adapter, one is a
PEFT LoRA adapter. **Every backbone is frozen.** No backbone is fine-tuned end-to-end; no backbone
weight is redistributed. Training produces a small module on top of a frozen encoder, and — for five
of the six artifacts — it does so by training on **cached embeddings or cached features**, never on raw
imagery decoded inside the training loop.

> **The single most important rule in this document: do not fabricate.** Every hyperparameter and every
> measured figure below was read from a file. Where a figure is a *quoted* value, the file it came from
> is named inline. Where a fact is **not** established from the available evidence, this document says
> so explicitly (`UNKNOWN — not established from the available evidence`) rather than estimating.

**The second most important rule:** a training run produces an **artifact**, not a **verified
capability**. The external training guide for the change-VQA head states this as a contract, verbatim:

> **Training produces an artifact, not a verified capability, and the run record says
> `TRAINED_UNVERIFIED`.** (`docs/R02_KAGGLE_TRAINING_GUIDE.md`)

Every artifact in this document carries a status. None of them is promoted to a system-level claim by
the fact that it trained. See [`BENCHMARKS.md`](BENCHMARKS.md) for the evaluation-side view and
[`MODELS.md`](MODELS.md) for the released-artifact view.

---

## Table of contents

1. [The training philosophy](#1-the-training-philosophy)
2. [Where each artifact trains — the reproducibility boundary](#2-where-each-artifact-trains--the-reproducibility-boundary)
3. [The frozen training configuration](#3-the-frozen-training-configuration)
4. [Router adapter — local CPU, cached embeddings](#4-router-adapter--local-cpu-cached-embeddings)
5. [Grounding head — local CPU, cached features](#5-grounding-head--local-cpu-cached-features)
6. [Change head — external GPU, STANet-style Siamese](#6-change-head--external-gpu-stanet-style-siamese)
7. [Optical-SAR fusion head — local CPU, seed sweep](#7-optical-sar-fusion-head--local-cpu-seed-sweep)
8. [Change-VQA head — external GPU, cached change features](#8-change-vqa-head--external-gpu-cached-change-features)
9. [VLM LoRA adapter — external GPU, PEFT](#9-vlm-lora-adapter--external-gpu-peft)
10. [Calibration — fitted, not trained](#10-calibration--fitted-not-trained)
11. [The reproducibility contract for training](#11-the-reproducibility-contract-for-training)
12. [What was NOT trained — exhaustive](#12-what-was-not-trained--exhaustive)
13. [What is NOT RUN / OPEN / BLOCKED for this topic](#13-what-is-not-run--open--blocked-for-this-topic)
14. [Where the evidence lives](#14-where-the-evidence-lives)

---

## 1. The training philosophy

### 1.1 Frozen backbone + small trainable head

The project's core architectural decision is that **the backbone is never updated**. Every artifact in
this document is a small module attached to an encoder that is loaded from a pinned revision and then
frozen. The frozen contracts are enumerated in `docs/ARCHITECTURE_FREEZE.md` §2:

| Component | Frozen backbone | What is trainable | Freeze reference |
|---|---|---|---|
| Intent router | `sentence-transformers/all-MiniLM-L6-v2` | the adapter only | `docs/ARCHITECTURE_FREEZE.md` §2.1 |
| VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | a LoRA delta on the language model | `docs/ARCHITECTURE_FREEZE.md` §2.2 |
| Grounding | `chendelong/RemoteCLIP` (`ViT-B-32`) | projection + grounding head | `docs/ARCHITECTURE_FREEZE.md` §2.3 |
| Change | — (reimplemented, not vendored) | the whole Siamese network (see §6) | `docs/ARCHITECTURE_FREEZE.md` §2.4 |
| Optical-SAR | `antofuller/CROMA` (`CROMA_base.pt`) | the fusion head only | `docs/ARCHITECTURE_FREEZE.md` §2.5 |
| Change-VQA | frozen STANet change detector | a two-stage 1.45 M-parameter head | `docs/R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §2 |

The **one exception** is the change head. `artifacts/change/levir_change_v001/model_metadata.json` and
the checkpoint's own embedded config record `frozen_encoder: false` and `pretrained_used: true`
(`artifacts/change/eval_test/eval_result.json` → `checkpoint_embedded_config`). The STANet detector's
ResNet-18 encoder is initialised from torchvision `IMAGENET1K_V1` and is **trained**, not frozen — which
is why its checkpoint is 63 MB while the other heads are 5–14 MB. This is a deliberate, recorded
exception, not an inconsistency: the change detector *is* the trained artifact, whereas the other five
artifacts sit on top of a backbone someone else trained.

### 1.2 Cached-embedding training — the cost-reduction strategy

Because the backbone is frozen, its output for a fixed input never changes. The project therefore
computes the backbone output **once**, writes it to a cache, and trains the head on the cache. This
turns a GPU-bound problem into a CPU-bound one and is the reason four of the six artifacts train on a
laptop:

| Artifact | What is cached | Cache artifact |
|---|---|---|
| Router | MiniLM sentence embeddings (384-d) | `artifacts/router/cache/` |
| Grounding | RemoteCLIP patch + text features | `artifacts/grounding/remoteclip_grounding_v001/cache/remoteclip_224_v1` |
| Optical-SAR | CROMA fused features (2318-d) | `artifacts/optical_sar/fusion_features/{train,val,test}.npz` |
| Change-VQA | change features (1045-d) + MiniLM question features (384-d) | `.../prepared/change_features.npz`, `.../prepared/text_features_<split>.npz` |
| VLM | **not cached** — the VLM LoRA adapter trains on rendered images | — |

The router is the clearest demonstration. `router/adapter.py` records, verbatim in its module docstring:

> Measured cost (Phase 4 probe): 50,822 parameters, 20 epochs over 4,096 x 384

i.e. **20 epochs over 4,096 cached 384-dimensional vectors**, on CPU. The measured wall-clock for that
probe is **0.28 s** (`router/adapter.py`). No GPU is required for any of the four CPU-trained
artifacts.

### 1.3 The frozen configuration hash

Every trained artifact records the frozen config hash it was trained against. The hash is computed over
the **parsed configuration object** — not over the file bytes — by `core/config.py:77-80`:

```
78f1e3700da15aa1
```

This value appears in `artifacts/change/levir_change_v001/model_metadata.json` →
`config_hash`, `artifacts/change_vqa/run/run_record.json` → `config_hash`,
`artifacts/optical_sar/fusion_head_v001/armA_seed100/run_record.json` → `config_hash`,
`artifacts/grounding/remoteclip_grounding_v001/run_record.json` → `config_hash`, and
`models/manifest.json` → `artifacts[*].config_hash`. It is a frozen invariant in
`docs/PHASE9_FREEZE.md` §3 and is enforced by
`tests/test_config.py::test_config_hash_matches_the_shipped_checkpoint_record`.

> **Editing `configs/base.yaml` moves the hash and detaches every artifact from it.** This is stated
> as a hazard, not a nicety, in `docs/PHASE9_FREEZE.md` §6: a config edit makes
> `scripts/eval_change.py` exit `3`. The change head's `head.pt` is *not* reachable through
> `change.checkpoint_path` (which resolves to `None`) precisely so that the benchmark number stays
> attached to an unmodified config; the wiring is done through a registry `builders=` override instead.

**A second hash exists and must not be confused with the first.** The router adapter records its *own*
adapter-config hash, `615478910dc266bf`, in `artifacts/router/router_adapter_v001/metadata.json` →
`config_hash` and `artifacts/router/threshold_sweep_val.json` → `adapter_config_hash`. That hashes the
adapter's config block, not the repo-wide config. `models/manifest.json` attaches the repo-wide
`78f1e3700da15aa1` to the released router artifact. Both are recorded; they hash different objects.

### 1.4 The hash-exempt environment channels

Some training runs need to vary a knob without editing `configs/base.yaml` — because editing the config
would move the hash. The project solves this with **environment channels that are deliberately excluded
from the hash**:

| Channel | Effect | Used by |
|---|---|---|
| `SATQUERY_CROMA_USE_8_BIT` | selects the optical-SAR preprocessing arm (`use_8_bit` true/false) | the optical-SAR seed sweep (§7) |
| `SATQUERY_DEVICE` | overrides device selection | training scripts generally |

`docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §3.1 records that the arm "travels on the
hash-exempt env channel, never through `configs/base.yaml`", which is why all ten optical-SAR run
records carry `config_hash: 78f1e3700da15aa1` (§7.7 of that document). `docs/REPRODUCIBILITY.md` §2.6
documents the same mechanism from the reproduction side.

### 1.5 No magic numbers in Python

Every hyperparameter lives in `configs/base.yaml`. Training modules read them from the config object;
they do not carry their own defaults for anything that affects a result. Where a module *does* carry a
constant (e.g. the change-VQA trainer's `SEED = 42`, `DEFAULT_EPOCHS = 40`,
`DEFAULT_TIME_LIMIT_SECONDS = 3*3600` in `training/change_vqa/train.py`), it is a CLI default that the
external run overrides explicitly and that is recorded in the run's `argv`.

### 1.6 What training never does

The frozen non-negotiables (`docs/ARCHITECTURE_FREEZE.md` §5) bound what training may do:

- **No LLM-generated coordinates.** The VLM is never used to produce coordinates.
- **No LLM-generated confidence.** Confidence comes from the evidence engine, not a language model.
- **No hidden-test-specific branches, thresholds, or prompts.** Public test is immutable; hidden data
  must never influence a threshold, a prompt or a routing decision.
- **Scene-level leakage isolation.** Splits are by scene (or group), never by sample.
- **Prompts are versioned files, frozen before benchmark evaluation.**
- **Every result carries an observable execution trace. No chain-of-thought.**

These are enforced in code, not by convention — see
[`DATASETS.md`](DATASETS.md) §7 for the leakage module and the public-test firewall.

---

## 2. Where each artifact trains — the reproducibility boundary

The single most consequential fact about this project's training story is that **two of the six
artifacts were trained outside this repository**, on an external GPU, and the repository ships the
*contract and the gate* for those runs rather than a one-command retrain.
`docs/REPRODUCIBILITY.md` §8 states this plainly.

| # | Artifact | Where it trains | Hardware | Guide / record |
|---|---|---|---|---|
| 1 | `router/adapter.pt` | **local CPU** | CPU only | `configs/base.yaml` §`router.training` |
| 2 | `grounding/head.pt` | **local CPU** | CPU (`torch 2.14.0+cpu`) | `scripts/train_grounding.py` |
| 3 | `change/head.pt` | **external GPU (Kaggle)** | Tesla T4 | `docs/PHASE9_GPU_HANDOFF.md` |
| 4 | `optical_sar/head.pt` | **local CPU** (seed sweep) | CPU (`torch 2.14.0+cpu`) | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| 5 | `change_vqa/head.pt` | **external GPU (Kaggle)** | **T4 ×2** | `docs/R02_KAGGLE_TRAINING_GUIDE.md` |
| 6 | `vlm/adapter_model.safetensors` | **external GPU (Kaggle)** | **T4 ×2** | `RUNBOOK_PHASE6_VLM_KAGGLE.md` |

> **The change head is trained on an external GPU, not locally.** Its `run_record.json` records
> `artifact_dir: /kaggle/working/satquery-ai/artifacts/change/levir_change_v001` and
> `data_root: /kaggle/input/datasets/keykeylv/levir-cd-256`
> (`artifacts/change/levir_change_v001/run_record.json`). The earlier `docs/PHASE9_GPU_HANDOFF.md`
> describes the handoff; `docs/PHASE9_GPU_RUN_RESULTS.md` reports the returned run.

### 2.1 Local CPU vs external GPU — what "local" means

"Local" in this document means the development machine recorded in the run records, e.g.
`artifacts/optical_sar/fusion_head_v001/armA_seed100/run_record.json` → `environment`:
`{"cuda_available": false, "platform": "Windows-10-10.0.26200-SP0", "python": "3.11.16",
"torch": "2.14.0+cpu"}`. The grounding run record records the same
(`artifacts/grounding/remoteclip_grounding_v001/run_record.json` → `environment`), as does the router
threshold sweep (`artifacts/router/threshold_sweep_val.json` → `environment`).

The four locally-trained artifacts therefore reproduce on a machine with **no CUDA device at all**.
This is a direct consequence of the cached-feature strategy in §1.2.

### 2.2 The mixed-precision (AMP) policy

The precision policy is frozen and has a hardware reason.

| Situation | Precision used | Reason |
|---|---|---|
| CUDA device present | **fp16** with `GradScaler` | `docs/ARCHITECTURE_FREEZE.md` §4: "T4 = SM 7.5, no bf16 tensor cores (C-6)" |
| CPU only | fp32 (autocast is a no-op) | no CUDA dispatch key to reach |
| bf16 requested | **refused** | `training/vlm/trainer.py` refuses bf16 (finding C-6) |

`training/vlm/config.py` defines `PRECISION_CHOICES = ("fp16", "bf16", "fp32")`, but the trainer
refuses the bf16 branch. The change-VQA trainer's AMP selector is more careful still: it gates on
compute capability rather than on `torch.cuda.is_bf16_supported()`, because **that predicate returns
`True` on a T4** via *emulated* bf16, which uses no tensor cores and is therefore slower than fp16 and
no more accurate. The correct branch on a T4 is **fp16 with a scaler**, and the first external Kaggle
log confirms the selector took it:
`mixed precision : True (cuda autocast (float16) with GradScaler)`
(`PRE_KAGGLE_READINESS_REPORT.md` §1b, K2). The recorded value is
`artifacts/change_vqa/run/run_record.json` → `optimization.amp_reason`.

**A defect class follows from this policy and is worth stating here.** Computing a binary
cross-entropy on *sigmoid probabilities* — `BCE(sigmoid(z), t)` — saturates: its backward is
`(p − t)/(p(1 − p))`, and at `p = 1.0` exactly PyTorch clamps that to a **finite** `−9.99999996e11`, so
`GradScaler` cannot flag it. Under fp16 autocast the cast node turns it into `−inf`, the sigmoid's own
backward multiplies by `sigmoid'(z) = 0`, and `inf × 0 = nan`. Saturation needs no AMP at all:
`sigmoid(17.0)` is exactly `1.0` in plain fp32. The fix is to compute the loss from the **pre-sigmoid
logits** with `binary_cross_entropy_with_logits`. The full chain, the reproduction table, and the seven
tests that pin it are in `PRE_KAGGLE_READINESS_REPORT.md` §1b (K3) and
`R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §4. The relevant source is
`training/change_vqa/model.py` (`binary_cross_entropy_from_logits`, and the docstring on
`binary_cross_entropy_probabilities` which survives as the clamped fallback).

### 2.3 Device selection

Device is resolved per run and recorded in every run record's `environment.requested_device`. No run
silently falls back: the change-VQA notebook "refuses to fall back silently" on a missing GPU
(`RUNBOOK_CHANGE_VQA_KAGGLE.md` §4), and the trainer's environment cell "stopped on `DEVICE is 'cuda'
but no GPU is visible`, which is the correct behaviour on this CPU-only machine"
(`PRE_KAGGLE_READINESS_REPORT.md` §6).

---

## 3. The frozen training configuration

`configs/base.yaml` is the single source of truth for every hyperparameter. It is a frozen artifact:
`docs/PHASE9_FREEZE.md` §2 records it at **10,637 bytes, SHA256 prefix `88434f7f8f78e2b8`, mtime
2026-09-16 19:30** — the *last* artifact to have moved, two days before the change-head GPU run. The
full freeze (every frozen key, with reason) is `docs/ARCHITECTURE_FREEZE.md` §4; the reproduction-side
account is [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §2.

The training-relevant blocks are reproduced here, key by key, from `configs/base.yaml`.

### 3.1 `router` — and `router.training`

| Key | Value |
|---|---|
| `model` | `sentence-transformers/all-MiniLM-L6-v2` |
| `revision` | `1110a243fdf4` |
| `max_length` | 128 (asserted ≤ 256) |
| `embedding_dim` | 384 |
| `device` | `auto` |
| `hidden_dim` | 128 |
| `dropout` | 0.10 |
| `confidence_threshold` | 0.70 |
| `num_tasks` | 6 |
| `tasks` | `[vqa, caption, grounding, change, optical_sar, unsupported]` |
| `training.epochs` | 60 |
| `training.batch_size` | 64 |
| `training.learning_rate` | 0.001 |
| `training.weight_decay` | 0.01 |
| `training.task_loss_weight` | 1.0 |
| `training.modality_loss_weight` | 0.3 |
| `training.binary_loss_weight` | 0.5 |
| `training.val_ratio` | 0.15 |
| `training.hard_negatives_to_test` | **true** |

### 3.2 `grounding_training` and `grounding_head`

| Key | Value |
|---|---|
| `grounding_training.learning_rate` | 0.0001 |
| `grounding_training.batch_size` | 16 |
| `grounding_training.epochs` | 20 |
| `grounding_training.weight_decay` | 0.0001 |
| `grounding_training.warmup_ratio` | 0.05 |
| `grounding_training.grad_clip` | 1.0 |
| `grounding_training.val_fraction` | 0.10 |
| `grounding_training.num_workers` | 2 |
| `grounding_training.save_every_steps` | 500 |
| `grounding_training.box_loss_weight` | 0.5 |
| `grounding_training.giou_loss_weight` | 0.3 |
| `grounding_training.confidence_loss_weight` | 0.2 |
| `grounding_head.feature_dim` | **2048** (enforced — see §5.2) |
| `grounding_head.positive_confidence_weight` | **20.0** |
| `grounding_head.decode` | `cell_relative` |

### 3.3 `change`

| Key | Value |
|---|---|
| `tile_size` | 256 |
| `tile_overlap` | 0 |
| `threshold` | 0.50 |
| `min_component_pixels` | 32 |
| `encoder` | `resnet18` |
| `sa_mode` | `PAM` |
| `learning_rate` | 0.001 |
| `batch_size` | 8 |
| `bce_weight` | 0.5 |
| `dice_weight` | 0.5 |
| `levir_split` | `{train: 7120, val: 1024, test: 2048}` |
| `checkpoint_path` | `None` (see §6.8 — deliberate) |

### 3.4 `croma` and `fusion`

| Key | Value |
|---|---|
| `croma.variant` | `base` |
| `croma.image_resolution` | 120 (native; `120 % 8 == 0` → 225 patches) |
| `croma.encoder_dim` | 768 |
| `croma.optical_channels` | 12 |
| `croma.sar_channels` | 2 |
| `fusion.input_dim` | **2318** = 3 × 768 + 12 + 2 (enforced — see §7.2) |
| `fusion.hidden_dim` | 512 |
| `fusion.dropout` | 0.2 |
| `fusion.num_classes` | 19 |

### 3.5 `training` — the VLM LoRA block

| Key | Value |
|---|---|
| `precision` | **fp16** |
| `vlm_batch_size` | 2 |
| `vlm_gradient_accumulation` | 8 |
| `vlm_learning_rate` | 0.0002 |
| `vlm_epochs` | 1 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `weight_decay` | 0.01 |
| `warmup_ratio` | 0.05 |
| `gradient_checkpointing` | true |
| `save_every_steps` | 500 |
| `vlm.processor_longest_edge` | **512** (frozen — finding F5-2) |

### 3.6 `evaluation` — the invariants that bound training

| Key | Value | Meaning |
|---|---|---|
| `immutable_public_test` | `true` | the public test split may not be re-scored or edited |
| `hidden_data_access` | `false` | hidden annotations are never available during development |
| `leakage_split_key` | `scene_id` | splits are keyed on scene, not sample |
| `official_aggregate_weights` | `null` | inventing an aggregate formula is prohibited |

`evaluation/leakage.py::assert_no_hidden_access` enforces the last two at runtime and *raises* if either
is violated.

### 3.7 `deployment` — frozen paperwork, not a training knob

`deployment.torch_compile: false` (ZeroGPU does not support it — C-8) and
`deployment.cpu_mode: required` are frozen because a training-time compile flag would change the
artifact's execution semantics. `docs/ARCHITECTURE_FREEZE.md` §4 records the reason for each. See
[`DEPLOYMENT.md`](DEPLOYMENT.md).

### 3.8 What invalidates the hash

Per `docs/REPRODUCIBILITY.md` §2.7, the hash moves when any *hashable* key changes. Two things do
**not** move it:

1. A value that travels on a hash-exempt environment channel (§1.4).
2. A call-site argument such as a registry `builders=` override — which is exactly why the change
   head's serving wiring and the change-VQA serving wiring both use that seam (§6.8, §8.10).

Everything else in the table above does. A moved hash detaches every artifact from its recorded
config and trips the drift guard.

---

## 4. Router adapter — local CPU, cached embeddings

### 4.1 What it is

A small classifier attached to the frozen MiniLM sentence encoder. It emits **five heads**:
`task` (6 classes), `modality` (4 classes), and three binary heads (`temporal`, `spatial_output`,
`language_output`). The architecture is fixed in `docs/ARCHITECTURE_FREEZE.md` §2.1:

```
embedding (384)
    |
LayerNorm
    |
Linear(384 -> hidden_dim)      default hidden_dim = 128
    |
GELU
    |
Dropout(0.1)
    |
    +--> task_head            Linear(hidden, 6)
    +--> modality_head        Linear(hidden, 4)
    +--> temporal_head        Linear(hidden, 1)   logit
    +--> spatial_head         Linear(hidden, 1)   logit
    +--> language_head        Linear(hidden, 1)   logit
```

`router/adapter.py` initialises every head with small-std init (`std = 0.02`) and zero bias, keeping the
initial sigmoid near 0.5 — without it the binary heads can start saturated and the BCE gradients vanish
before the task head learns anything. `forward` asserts the input is 2-D and that
`embeddings.shape[1] == input_dim`, and requires embeddings already detached from the frozen encoder:
the adapter does not back-propagate into MiniLM.

**Parameter-count discrepancy — flagged, not smoothed over.** Two figures exist and have not been
reconciled:

| Figure | Source |
|---|---|
| **~50,822** | `router/adapter.py` module docstring ("Phase 4 probe"); `models/manifest.json` architecture string |
| **51,725** | `artifacts/router/router_adapter_v001/metadata.json` → `num_parameters`; `docs/PHASE4_ROUTER_REPORT.md` |

The released artifact is **211,961 bytes** (`models/manifest.json`). Which figure is authoritative is
`UNKNOWN — not established from the available evidence`. This document quotes both with their sources
and does not pick one. (`docs/MODELS.md` §3.5 records the same discrepancy.)

### 4.2 The corpus

The router is trained on a **synthetic** corpus — hand-written plus template-generated queries. It is
small, and every artifact says so.

| Property | Value | Source |
|---|---|---|
| total examples | **576** | `artifacts/router/router_adapter_v001/metadata.json` → `corpus.total` |
| groups | **54** | same → `corpus.groups` |
| by source | curated **66**, template **510** | same → `corpus.by_source` |
| corpus hash | `8054810736ef97c3db873b2d7073948773a8982f1d16411833d39e15e1871e83` | same → `corpus.hash` |
| by task | caption 91, change 115, grounding 128, optical_sar 50, unsupported 105, vqa 87 | same → `corpus.by_task` |
| binary positives | language_output 471, spatial_output 164, temporal 115 | same → `corpus.positives` |

The plan's minima are `plan_min_val_queries: 500` and `plan_min_hard_negatives: 100`
(`artifacts/router/threshold_sweep_val.json`). The corpus is **below both**, and the artifact's own
note says so verbatim:

> *"corpus-limited: val n=86 vs plan >=500. This is NOT a calibration — the corpus is synthetic and too
> small (min per-class support 8, caption) and val carries 0 hard negatives (hn_* families are held out
> to TEST by design). Selecting a threshold here yields a justified default, not a calibrated value. The
> corpus was NOT padded with generated queries."*

### 4.3 The frozen-encoder + cached-embedding strategy

This is the artifact where the strategy is most visible. The encoder is frozen, so its 384-d output for
a fixed string is constant; the project embeds the 576-example corpus once, caches the vectors under
`artifacts/router/cache/`, and trains the adapter on the cache. The recorded cost is the probe figure
quoted in §1.2: **20 epochs over 4,096 × 384 vectors in 0.28 s on CPU**.

The whole router run took **4.92 s** wall-clock
(`artifacts/router/router_adapter_v001/metadata.json` → `duration_seconds`), and the threshold sweep
that follows training took **0.206 s** (`artifacts/router/threshold_sweep_val.json` → `seconds`). There
is no GPU in either record.

### 4.4 Group-based splits with hard negatives in the test split

The router is the artifact where the split rule is most explicit, and it is a **leakage-prevention**
rule rather than a convenience.

- Splits are **by group** — template family / hard-negative family — **never by example**
  (`router/dataset.py::split_by_group`).
- The reason, stated in `docs/ARCHITECTURE_FREEZE.md` §2.1's neighbourhood and in
  [`MODELS.md`](MODELS.md) §3.5: template-generated queries are near-duplicates. Splitting by example
  would put `"Show me the water body."` in train and `"Show me the road."` in val — one token apart —
  and report a fake accuracy.
- **Hard-negative families are placed in the test split.** `configs/base.yaml` sets
  `router.training.hard_negatives_to_test: true`, and `router/dataset.py` prefixes hard-negative
  families with `hn_`. Finding **F4-3** records the consequence: hard-negative families are held out to
  test so their accuracy measures **generalisation rather than memorisation**.

**Measured split sizes** (`artifacts/router/router_adapter_v001/metadata.json` → `split`):

| Split | examples | groups | hard negatives |
|---|---|---|---|
| train | 410 | 37 | — |
| val | 86 | 8 | **0** (by design) |
| test | 80 | 9 | the `hn_*` families |

The split audit records `clean: true` and `groups_across_splits: {}` — i.e. no group straddles two
splits.

### 4.5 Hyperparameters

From `configs/base.yaml` §`router.training`, corroborated by
`artifacts/router/router_adapter_v001/metadata.json`:

| Hyperparameter | Value |
|---|---|
| epochs | 60 |
| batch size | 64 |
| learning rate | 0.001 |
| weight decay | 0.01 |
| task loss weight | 1.0 |
| modality loss weight | 0.3 |
| binary loss weight | 0.5 |
| `val_ratio` | 0.15 |
| `hard_negatives_to_test` | true |
| seed | 42 |

### 4.6 The full training procedure

1. **Embed** the 576-example corpus with the frozen MiniLM encoder at `max_length: 128` and cache the
   384-d vectors.
2. **Split by group** (`split_by_group`), holding `hn_*` families out to test.
3. **Train the adapter** for 60 epochs, batch 64, AdamW-class optimisation at `lr = 0.001`,
   `weight_decay = 0.01`, on the composite loss
   `1.0 · task + 0.3 · modality + 0.5 · binary`.
4. **Select** the epoch by validation (group-split) performance; the history is recorded per epoch in
   `artifacts/router/router_adapter_v001/metadata.json` → `history` (each entry carries `epoch`, `loss`,
   `lr`, `val_combined_accuracy`, `val_task_accuracy`).
5. **Sweep the confidence threshold** on **val only** (`scripts/` sweep → 
   `artifacts/router/threshold_sweep_val.json`), 50 thresholds from 0.50 to 0.99.
6. **Ship** the adapter (`artifacts/router/router_adapter_v001/adapter.pt`).

### 4.7 Measured numbers

`artifacts/router/threshold_sweep_val.json` is the release's router artifact. Its headline:

| Field | Value | Key path |
|---|---|---|
| **overall ungated task accuracy** | **0.965116** | `overall_ungated_accuracy` |
| split | `val` | `split` |
| n val | **86** | `n_val` |
| n val examples scored | 86 | `n_val_examples_scored` |
| **n test examples scored** | **0** | `n_test_examples_scored` |
| **`test_split_touched`** | **false** | `test_split_touched` |
| `corpus_limited` | true | `corpus_limited` |
| hard negatives in val | **0** | `hard_negatives_in_val` |
| val min per-class support | 8 (caption) | `val_min_support` |
| split sizes | train 410 / val 86 / test 80 | `split_sizes` |
| adapter encoder params | 22,713,216 | `adapter_encoder.parameters` |
| adapter config hash | `615478910dc266bf` | `adapter_config_hash` |
| repo config hash | `78f1e3700da15aa1` | `config_hash` |

> ### **0.965116 is validation-only, ungated, n = 86. The router TEST split was NOT RUN.**
>
> `n_test_examples_scored` is **0** and `test_split_touched` is **false**. Do not read 0.965116 as a
> test result. The number is "ungated" because the plan's acceptance target (`docs/` §59 in the master
> plan: task accuracy ≥ 95 % on 500 validation queries / 100 hard negatives / 50 unsupported) is
> measured on a corpus that has 86 val examples and **0** hard negatives. A router can reach high
> accuracy on easy queries while failing exactly on the hard-negative families the plan calls out.

**Historical note, recorded so the two are not conflated.** `artifacts/router/router_adapter_v001/
metadata.json` → `metrics.test` contains a **test block** (n = 80, `task_accuracy: 0.975`,
`macro_f1_task: 0.976`, `hard_negative_accuracy: 0.8`). That block is the **earlier Phase 4 gate-2
evaluation**, which predates the shipped threshold sweep and carries its own caveat — per
`docs/PHASE4_ROUTER_REPORT.md`, *"the 0.975 headline is partly earned on templates the split kept in
training. Treat the router as working, not as benchmarked."* The release's position for the shipped
artifact is **TEST NOT RUN**; the Phase 4 numbers are retained as a historical record, not promoted to
a release benchmark. (`docs/MODELS.md` §3.5 records the same distinction.)

### 4.8 The threshold sweep is val-only by construction

`artifacts/router/threshold_sweep_val.json` iterates thresholds 0.50 → 0.99 and records `coverage`,
`covered_task_accuracy`, `fallback_rate` and `n_covered` at each.

| Row | Threshold | coverage | covered_task_accuracy | fallback_rate | n_covered |
|---|---|---|---|---|---|
| shipped (`shipped_row`) | **0.70** | 0.848837 | 0.972603 | 0.151163 | 73 |
| sweep-selected (`selected`) | 0.76 | 0.790698 | **1.0** | 0.209302 | 68 |

`select_by: "covered_accuracy"`. The delta against the shipped threshold
(`delta_vs_shipped`) is `coverage: −0.0581`, `covered_task_accuracy: 0.0274` — the sweep's own criterion
trades **5.8 pp of coverage for 2.7 pp of covered accuracy**. **The shipped threshold remains 0.70.**
This is a *justified default*, not a calibrated value, and the artifact says so.

---

## 5. Grounding head — local CPU, cached features

### 5.1 What it is

A trainable head over the **frozen** RemoteCLIP `ViT-B-32` encoder
(`chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt`, 605 MB). The encoder is frozen; the projection
and the grounding head are trainable (`docs/ARCHITECTURE_FREEZE.md` §2.3). The head has **1,052,677
parameters** (`artifacts/grounding/remoteclip_grounding_v001/training_metadata.json` →
`head_parameters`) and emits five outputs — `tx, ty, tw, th, objectness` — over a **7 × 7** token grid
(`grid: 7`, `head.decode: cell_relative`, `head.dropout: 0.1`, `head.feature_dim: 2048`,
`head.hidden_dim: 512`).

### 5.2 The enforced 2048-dimensional invariant

Per-cell feature is the concatenation

```
concat([patch, text, patch·text, global_pool]) = 4 × 512 = 2048
```

`core/config.py` **rejects any value other than `4 × grounding.encoder_projected_dim` at load time**, and
the specialist asserts the same 512 against the real model. The reason is stated in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §2.3 (invariant 2): a mismatch is a **silent** shape error
that torch only raises at the similarity step — after patch features are already cached. The load-time
guard makes the error loud and early. `configs/base.yaml` records `grounding_head.feature_dim: 2048`.

### 5.3 The feature cache

The grounding run is a **two-stage** pipeline (`scripts/train_grounding.py`): stage 1 extracts and caches
RemoteCLIP features, stage 2 trains the head on the cache. The recorded cache
(`artifacts/grounding/remoteclip_grounding_v001/run_record.json` → `cache`) is:

| Quantity | Value |
|---|---|
| cache dir | `artifacts/grounding/remoteclip_grounding_v001/cache/remoteclip_224_v1` |
| images total | 15,699 |
| image cache hits / misses | 15,699 / 0 |
| images undecodable / missing source | 0 / 0 |
| phrases total | 24,439 |
| text cache hits / misses | 24,439 / 0 |
| cache build seconds | 2.1 (images 0.9 s, text 0.0 s) |

The cache version string is `remoteclip_224_v1` — the resolution is baked into the cache identity, so a
head trained at 224 cannot silently be served features extracted at another resolution.

### 5.4 Hyperparameters

From `configs/base.yaml` §`grounding_training`, corroborated by the run's history (first epoch
`lr = 0.0001`, decaying):

| Hyperparameter | Value |
|---|---|
| learning rate | 0.0001 |
| batch size | 16 |
| epochs | 20 |
| weight decay | 0.0001 |
| warmup ratio | 0.05 |
| grad clip | 1.0 |
| val fraction | 0.10 |
| num workers | 2 |
| save every steps | 500 |
| box loss weight | 0.5 |
| GIoU loss weight | 0.3 |
| confidence loss weight | 0.2 |
| `positive_confidence_weight` | **20.0** |
| seed | 42 |

The loss weights appear in the per-epoch history as `loss_box`, `loss_giou`, `loss_confidence` and
`loss_total` (`artifacts/grounding/remoteclip_grounding_v001/run_record.json` → `history`).

### 5.5 Why `positive_confidence_weight = 20.0`

The objectness branch is a per-cell binary classification over a 7 × 7 grid. Roughly **1 cell in 49** is
positive. Unweighted, the loss-minimising solution is "no object everywhere"; the weight is what stops
that collapse. The value is frozen in `configs/base.yaml` §`grounding_head`, not chosen per run.

### 5.6 The training procedure

1. **Extract** RemoteCLIP image + text features for the VRSBench grounding split, into
   `cache/remoteclip_224_v1` (§5.3).
2. **Split** by image (leakage split by image; `training/grounding/dataset.py`), with
   `val_fraction: 0.10`. `run_record.json` records `train_items: 23042` / `val_items: 2548`;
   `training_metadata.json` records `train_images: 14130` / `val_images: 1569`; and the cache's
   `images_total` is 15,699 = 14,130 + 1,569. `docs/PHASE8_HANDOFF.md` §2 restates the same split.
3. **Train** the 1,052,677-parameter head for 20 epochs, batch 16, `lr = 1e-4`, `weight_decay = 1e-4`,
   `warmup_ratio = 0.05`, `grad_clip = 1.0`, on
   `0.5 · box + 0.3 · giou + 0.2 · confidence` with `positive_confidence_weight = 20.0`.
4. **Select** the best validation IoU. `docs/PHASE8_GROUNDING_HEAD_DECISION.md` records the decision.
5. **Evaluate** under two protocols and two decode variants (§5.9).
6. **Ship** `artifacts/grounding/remoteclip_grounding_v001/head.pt` (12,639,041 bytes, sha256
   `93432f7034be91a8ffd9c1a84e3eeec00bed7832c043fe7f83d2be230284c6bb`; `models/manifest.json`).

**Wall-clock:** **805.9 s** on CPU (`artifacts/grounding/remoteclip_grounding_v001/run_record.json` →
`wall_seconds`; the per-epoch `seconds` in the run history sum to 765.9 s over 20 epochs — ~38 s/epoch —
with the remainder being setup).

### 5.7 Resolution is frozen at 224 — 448 was REJECTED

`grounding.image_size: 224` (native). The 448 variant (→ 196 tokens) was a **gated experiment** in
`docs/ARCHITECTURE_FREEZE.md` §2.3: adopted only if validation IoU improved. It did not.

`docs/ARCHITECTURE_FREEZE.md` §4 records the resolution as "**RESOLVED by measurement**: 448 lost on
IoU, all recall thresholds and latency over 16,159 records (`docs/PHASE7_RESOLUTION_DECISION.md`)". The
paired test statistic is recorded as **mean diff −0.0147, 95 % CI [−0.0160, −0.0134], t = −22.63, at
1.59× latency** (see [`BENCHMARKS.md`](BENCHMARKS.md) §4.2 and
[`EVALUATION.md`](EVALUATION.md) §5, "a pre-registered rejection"). The localisation floor at 224 is
**32 px** (7 × 7 tokens of 32 px each).

### 5.8 Measured training numbers

From `artifacts/grounding/remoteclip_grounding_v001/training_metadata.json` and `run_record.json`:

| Field | Value | Key path |
|---|---|---|
| best val IoU | **0.0946** | `best_val_iou` (full: `0.09463294948954619` in `run_record.json`) |
| zero-shot baseline | **0.0972** | `baseline.mean_best_iou` |
| `beats_baseline` | **false** | `beats_baseline` |
| required margin | 0.02 | `baseline.required_margin` |
| first epoch val IoU (argmax decode) | 0.0436 | `history[0].val_iou_argmax` |
| head parameters | 1,052,677 | `head_parameters` |
| grid | 7 | `grid` |
| device | cpu | `device` |
| data root | `training/data/vrsbench` | `data_root` |

> **The head did not beat the zero-shot baseline on validation IoU** (0.0946 < 0.0972;
> `beats_baseline: false`, improvement −0.002567). This is a **negative result**, preserved as such.
> It does not invalidate the head — the head is what the two-protocol evaluation is computed with, and
> the zero-shot number (0.0972) is one of the two decode variants the release quotes. But no claim of
> improvement over zero-shot is made, because the measurement says there is none.

### 5.9 Two protocols × two decode variants

Grounding is the artifact where collapsing protocols is most dangerous, because a box-convention
mistake is silent. The release therefore reports:

| Protocol / variant | Value | Artifact |
|---|---|---|
| **canonical** protocol, `head_threshold` decode | **0.2838** | `eval_result_canonical.json` → `head_threshold.mean_best_iou` |
| **canonical** protocol, `head_argmax` decode | **0.1215** | `eval_result_canonical.json` → `head_argmax.mean_best_iou` |
| **matched6** protocol, `head_threshold` decode | **0.2566** | `eval_result_matched6.json` → `head_threshold.mean_best_iou` (top_k 6) |
| zero-shot baseline | **0.0972** | `run_record.json` → `baseline_mean_best_iou` |

`n_eval_records: 16159`, `resolution: 224` (`eval_result_canonical.json`). **Never quote one protocol
alone.** The convention is stated in [`BENCHMARKS.md`](BENCHMARKS.md) §1 and
[`EVALUATION.md`](EVALUATION.md) §4.2.

---

## 6. Change head — external GPU, STANet-style Siamese

### 6.1 What it is

A reimplemented STANet-style Siamese change detector — **reimplemented, not vendored**
(`docs/ARCHITECTURE_FREEZE.md` §2.4, finding C-9). Architecture
(`specialists/change/stanet.py`, and the checkpoint's embedded config in
`artifacts/change/eval_test/eval_result.json` → `checkpoint_embedded_config`):

| Component | Value |
|---|---|
| encoder | ResNet-18 (torchvision `IMAGENET1K_V1`, `pretrained_used: true`) |
| encoder channels | `[64, 128, 256, 512]` |
| self-attention | **PAM** (`sa_mode: PAM`) — the BAM alternative is not used |
| width | 128 |
| `frozen_encoder` | **false** (see §1.1 — this is the one trained encoder) |
| `attention_budget_bytes` | 268,435,456 (256 MiB; PAM is skipped at layer 1 to stay inside it) |
| tied weights | yes (Siamese) |
| model parameters | **15,779,969** |

### 6.2 Hyperparameters

From `configs/base.yaml` §`change`, corroborated by
`artifacts/change/levir_change_v001/run_record.json` → `hyperparameters`:

| Hyperparameter | Value |
|---|---|
| tile size | 256 |
| tile overlap | 0 |
| learning rate | 0.001 |
| batch size | 8 |
| epochs | 20 |
| weight decay | 0.0001 |
| grad clip | 1.0 |
| loss | `0.5 · BCE + 0.5 · Dice` (`bce_weight` 0.5, `dice_weight` 0.5) |
| `pos_weight` | `null` (not used) |
| LR schedule | cosine (`CosineAnnealingLR(optimizer, T_max = epochs)`, `training/change/train.py:750`) |
| threshold | 0.50 |
| `min_component_pixels` | 32 |
| seed | 42 |

### 6.3 Training data and split

LEVIR-CD-256, flat layout. `run_record.json` records `train_items: 7120`, `train_scenes: 445`,
`val_items: 1024`, `val_scenes: 64`; `docs/PHASE9_REAL_DATA_VERIFICATION.md` records the full split as
**445 / 64 / 128 scenes = 7120 / 1024 / 2048 tiles**, matching `change.levir_split`. The split is
scene-disjoint (verified — see [`DATASETS.md`](DATASETS.md) §3).

### 6.4 The procedure

1. **Handoff** the code and data to an external GPU. `docs/PHASE9_GPU_HANDOFF.md` records the Kaggle
   code zip `artifacts/kaggle/satquery-code.zip` (152 files, 0.6 MB), the cell map, and the cell-5
   gate.
2. **Train** 20 epochs, batch 8, `lr = 1e-3`, `0.5·BCE + 0.5·Dice`, cosine schedule.
3. **Select** the best validation IoU. `best_val_iou` full precision `0.8232294319484017`
   (`run_record.json`), `first_val_iou` `0.7335963646366181`; `improved: true`.
4. **Score the immutable test split exactly once.**
5. **Freeze** the phase (`docs/PHASE9_FREEZE.md`).

**Run identity:** `run_id: change_head_20260918T070723Z`, `wall_seconds: 4014.3`
(`run_record.json`) / `duration_seconds: 4012.29` (`training_metadata.json`). `peak_vram_mb: 1483.0`
(`training_metadata.json`). Device `cuda`, torch `2.10.0+cu128`, python 3.12.13.

### 6.5 Measured training numbers

From `artifacts/change/levir_change_v001/training_metadata.json` (and `model_metadata.json`, which
carries the identical block):

| Field | Value |
|---|---|
| best val IoU (pooled) | **0.8232** |
| final val pooled | f1 0.9030, iou 0.8232, miou 0.9075, precision 0.9176, recall 0.8890 |
| final val macro | f1 0.8317, iou 0.7507, miou 0.8645, precision 0.8778, recall 0.8122 |
| val n | 1024 |
| val images with change | 436 |
| val mean change fraction | 0.042 |
| first val IoU | 0.7336 |
| device | cuda |

### 6.6 The test result — the only `VERIFIED` headline

`artifacts/change/eval_test/eval_result.json`, `split: test`, `n: 2048`:

| Metric | pooled | macro |
|---|---|---|
| f1 | **0.8964** | 0.7962 |
| iou | **0.8122** | 0.7180 |
| miou | 0.9007 | **0.8457** |
| precision | 0.9195 | 0.8506 |
| recall | 0.8745 | 0.7757 |
| tp / fp / fn / tn | 5,978,997 / 523,658 / 858,407 / 126,856,666 | same pixel counts |
| n pixels | 134,217,728 | same |

`threshold: 0.5`, `tile_size: 256`, `seconds: 55.359`, `n_images_with_change: 935`,
`mean_change_fraction: 0.0509`, `config_drift: false`.

> **This is the only `VERIFIED` headline in the release** — pooled IoU 0.8122 / pooled F1 0.8964, with
> the macro pair (iou 0.7180 / **miou 0.8457**) reported alongside. `docs/PHASE9_FREEZE.md` §4 lists it
> as "Benchmark performance ✅ ... reproduced". The generalisation gap is val 0.8232 → test 0.8122.

**Metric-naming caveat.** The `macro` block carries *both* `iou` (0.7180, the macro-average of the
per-image IoU) and `miou` (0.8457, the mean IoU). The headline "macro IoU 0.8457" used across the
release docs is the **`miou`** field. This document names the key path so the two cannot be conflated.

### 6.7 The threshold lever is CLOSED

`docs/PHASE9_FREEZE.md` §5 eliminates the threshold hypothesis. A **val-only** sweep scored 19
thresholds; `scripts/sweep_change_threshold.py` refuses any `--split` other than `val` and exits `4`
before loading config, checkpoint, or data — so the sweep cannot touch the test split. Its artifact
records `test_split_touched: false` and its sha256 is
`34e20f62bc1dd7810f5ef5f26838213d59368cf1352a804d38cb0e140eaf97f2`.

| Threshold | pooled IoU | macro IoU |
|---|---|---|
| **0.50 (retained)** | 0.8232 | 0.7507 |
| best pooled (0.35–0.40) | 0.8239 (+0.0007) | — |
| best macro (0.25) | — | 0.7550 (+0.0043) |

Four thresholds do dominate 0.50 on all three reported metrics — but the last five validation epochs
span 0.8213–0.8232, a spread of **0.0019**, so the entire available threshold gain is **0.37× the
epoch-to-epoch noise**. A gain smaller than the run's own variance is not a finding. **0.50 is retained;
this hypothesis is eliminated, not deferred.**

### 6.8 The freeze, and the orphaned-head hazard

`docs/PHASE9_FREEZE.md` freezes the change specialist: the artifacts in its §2 may not be edited,
regenerated, or re-scored, and a longer cosine run must use `levir_change_v002` or it overwrites the
audited `head.pt` in place. The freeze also documents a verified hazard: the Kaggle notebook
`notebooks/kaggle_change_training.ipynb` **hardcodes the frozen output directory**, so re-running it
writes straight over `levir_change_v001/head.pt`.

The trained head was, at freeze time, **not reachable from the serving path**: `change.checkpoint_path`
resolves to `None`, so the registry builds a random-initialised detector and the change specialist
reports `DEGRADED` with `has_checkpoint: False`. This is deliberate — pointing `base.yaml` at the
checkpoint *moves the config hash* and detaches the benchmark number. The fix uses the registry
`builders=` override, which reaches the change builder without moving the hash
(`app/serving.py`; `docs/ARCHITECTURE_CHANGE_CHANGE_SERVING_WIRING.md`). Wired that way, a real
`AnalysisController.run()` on `test_79_2.png` returns `available`, `degraded=False`, confidence 0.7987.
Unwired, the same query returns `DEGRADED` with zero change regions. **"Phase 9 benchmark 0.8122" and
"the app detects change" are different claims, and only the first is true.**

### 6.9 The registration-gate defect — `OPEN`

`docs/PHASE9_FREEZE.md` §7 records an open defect inherited by Phase 10: the change specialist's
registration gate produces false positives. **1,202 / 2,048 test tiles (58.7 %)** are flagged as
mis-registered; `response < 0.15` is involved in 98.7 % of them; the flag rate climbs 42 % → 75 % → 98 %
→ 100 % as ground-truth change fraction rises. On 638 flagged tiles that do contain change, **5,679
region claims are suppressed** — 5,338 of them on tiles the model scores at IoU ≥ 0.7 (median 0.841).
An independent NCC check found **0 / 30** credible large offsets on the tiles the gate blames. One
counter-hypothesis remains untested: ~50 % of *zero-change* tiles are also flagged, consistent with low
image texture. **`OPEN`.** This is a *serving-path* defect; it does not affect the 0.8122 benchmark,
which is computed offline.

---

## 7. Optical-SAR fusion head — local CPU, seed sweep

### 7.1 What it is

A fusion head over the **frozen** CROMA-base encoder (`antofuller/CROMA`, `CROMA_base.pt`, MIT;
777,563,846 bytes; sha256 `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`). The head
is `LayerNorm → Linear(2318, W) → GELU → Dropout → Linear(W, task_dim)`
(`docs/ARCHITECTURE_FREEZE.md` §2.5), with `W = 512`, `task_dim = 19`, dropout 0.2, and **1,201,711
parameters** (`artifacts/optical_sar/fusion_head_production_v001/production_head_record.json` →
`head_config.head_parameters`).

### 7.2 The enforced 2318-dimensional invariant

The fusion head's input is derived, not hardcoded (`docs/REPRODUCIBILITY.md` §2.3, invariant 1):

```
optical_GAP      (B, 768)
SAR_GAP          (B, 768)
joint_GAP        (B, 768)
optical_mask     (B, 12)     <- availability, from sensor adapter
sar_mask         (B,  2)     <- availability, from sensor adapter
                 ---------
concat           (B, 2318)
```

`core/config.py` rejects any other value at load time. `docs/ARCHITECTURE_FREEZE.md` §2.5 records that
**CROMA never receives a mask** (finding C-1): the masks are *inputs to the fusion head*, which is what
lets the head learn to trust the availability signal. `croma.image_resolution: 120` is native
(`120 % 8 == 0` → 225 patches).

### 7.3 The feature caches

Training consumes cached CROMA features. The Arm-A cache
(`artifacts/optical_sar/fusion_features/`) contains
`train.npz` (170,919,398 B), `train.json` (1,581,540 B), `val.npz` (34,195,674 B),
`val.json` (317,536 B), `test.npz` (34,165,540 B), `test.json` (317,537 B). The Arm-B cache lives in
`artifacts/optical_sar/fusion_features_armB/`. These caches are **reproducible and are not released as
model weights**.

**The arm is baked into the cache.** `scripts/train_fusion.py` refuses to train a cache whose recorded
arm differs from `--arm`; the resume provenance guard
(`training/fusion/extract.py`, `CacheProvenanceError`) refuses to append Arm-B rows into an Arm-A cache.
This is why §7.4's blocker is a *compute* item, not a code defect.

### 7.4 The Arm A / Arm B contract — and why "Arm B" is not the registered Arm B

`docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §2 (D-01) amends the Phase 14 contract. The arms as
actually implemented and run:

| Arm | Definition | `use_8_bit` |
|---|---|---|
| **Arm A (control)** | percentile/dB conditioning → per-channel `mean ± 2·std` stretch → uint8 round-trip | **true** |
| **Arm B (variant)** | percentile/dB conditioning → per-channel `mean ± 2·std` stretch → bounded stretch, **no** uint8 round-trip | **false** |

Three facts must be stated together, or the pre-registration is silently rewritten:

1. **The §4-registered arm B — "percentile/dB only, no encoder-input stretch" — is NON-CONSTRUCTIBLE.**
   `normalise_for_croma` has no skip branch: it always applies the `mean ± 2·std` stretch, and
   `use_8_bit` only toggles the uint8 round-trip vs a clip
   (`specialists/optical_sar/radiometry.py:377-401`). The registered arm is retained unedited for the
   record and was never run.
2. **The arm actually run under the name "B" is the registered arm C** ("A but `use_8_bit=false`").
   Calling the current B the registered B would equate two different arms.
3. **No Arm C was introduced.** The arm set is frozen at exactly two.

So the comparison that was actually run is: **"does the uint8 round-trip in the encoder-input path
change fusion-head validation accuracy?"** — the quantisation/range-bound question — **not** the
stretch-vs-no-stretch question §4 originally registered. The deviation is recorded, not hidden.

The Arm-A run records carry a `normalization` string describing exactly what was executed, e.g.
`artifacts/optical_sar/fusion_head_v001/armA_seed100/run_record.json` →
`normalization: "uint8 axis, use_8_bit=true: per-channel mean+-2std encoder-input stretch, then the
uint8 round-trip (scale to 0..255, clip, quantise, /255) -> {0/255,...,1}. This is the transform
actually executed; the registered PHASE14 §4 stage-1 percentile/dB conditioning is not run."`

### 7.5 Hyperparameters

From `artifacts/optical_sar/fusion_head_v001/armA_seed100/run_record.json` (`hyperparameters`,
`head_config`), identical across both arms:

| Hyperparameter | Value |
|---|---|
| batch size | 64 |
| epochs | 20 |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| grad clip | 1.0 |
| warmup ratio | 0.05 |
| `input_dim` | 2318 |
| `hidden_dim` | 512 |
| `task_dim` | 19 |
| dropout | 0.2 |
| head parameters | 1,201,711 |
| seed | 100–104 |

### 7.6 Channel / band dropout is mandatory

`docs/ARCHITECTURE_FREEZE.md` §2.5: "Channel/band dropout during fusion-head training is **mandatory**;
it is what teaches the head to trust the availability mask." The recorded rates are

| Modality | dropout rates |
|---|---|
| optical | `[1.0, 0.8, 0.6, 0.4]` |
| SAR | `[1.0, 0.5]` |

(`artifacts/optical_sar/fusion_head_v001/armA_seed100/run_record.json` → `channel_dropout`). A rate of
`1.0` means the modality is entirely dropped for that sample, which forces the head to produce a
prediction from the surviving modality plus the mask.

### 7.7 The seed sweep and the decision

Training was run as **two arms × five seeds (100–104)**. The **deciding metric is
`best_val_accuracy`** — the maximum validation accuracy over epochs (`Gate F D-02`); macro-F1 is
co-reported, never deciding; `final_val` is not deciding.

| | Arm A | Arm B |
|---|---|---|
| accuracies | 0.83425 · 0.82450 · 0.83850 · 0.85300 · 0.83525 | 0.84300 · 0.84675 · 0.84450 · 0.84400 · 0.81725 |
| **mean** | **0.837100** | **0.839100** |
| min / max | 0.82450 / 0.85300 | 0.81725 / 0.84675 |
| spread (max − min) | **0.02850** | 0.02950 |

(`artifacts/optical_sar/fusion_head_v001/armA_seed_variance_report.json`,
`armB_seed_variance_report.json`; `docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §7.1.)

**The decision rule was pre-registered.** Gate F §6 (written *before* any Arm-B result existed) pinned
the A/B statistic to the five-seed **MEAN**:

```
B_mean − A_mean = 0.839100 − 0.837100 = +0.002000
threshold (fixed in §5/§6, never re-chosen) = 0.0285
+0.002000 > 0.0285  →  FALSE
```

> ### **ARM A RETAINED. Arm B is NOT adopted.**

The substantive finding: **the uint8 round-trip is not accuracy-limiting at this scale.** Removing it
shifts validation accuracy by **+0.2 percentage points** against a **2.85-point** seed-to-seed noise
floor — the effect is roughly **14× smaller than the noise it would have to clear**. Robustness checks
(§7.4 of the Gate F record) confirm the verdict does not depend on which arm supplies the floor, and
that under the excluded best-single-seed reading Arm A wins too. The paired design is proven: both arms
carry **identical `sample_ids` and `scene_ids` in identical order** while the `.npz` sha256 differs in
all three splits — same samples, same order, different bytes, so the delta is attributable to
preprocessing, not sampling.

**The σ = 0 fallback.** Gate F D-03 amends the decision rule so that if the measured spread is zero the
floor is the validation-accuracy resolution `1/N_val`. With `N_val = 4000`, that floor is **0.00025**.
It did not apply here (the measured spread is non-zero at 0.0285), but it is part of the contract.

### 7.8 The production head

The production head is a **distinct, frozen artifact** — not the same directory as the sweep.

`artifacts/optical_sar/fusion_head_production_v001/production_head_record.json`:

| Field | Value |
|---|---|
| designated by | **R-14** (owner ruling, 2026-09-21) |
| source | `artifacts/optical_sar/fusion_head_v001/armA_seed103/head.pt` |
| arm / seed | **A / 103** |
| sha256 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` |
| bytes | 14,427,457 |
| `best_val_accuracy` | **0.853** |
| `arm_a_mean_best_val_accuracy` | 0.8371 |
| `test_split_used_for_selection` | **false** |
| `is_ab_deciding_statistic` | **false** |

Selection basis, verbatim from the record: arm selection is the **five-seed MEAN** of
`best_val_accuracy` (pre-registered); production-head selection is the **highest `best_val_accuracy`
among the retained Arm-A seeds** (a post-experiment artifact-selection rule only). The five Arm-A
candidate digests are listed in `retained_arm_a_candidates`. The provenance note states the production
head bytes are an **unmodified, byte-identical copy** (`shutil.copyfile`) of the trained Arm-A seed-103
artifact — no re-serialisation, re-pickling or tensor round-trip. **The test split was not used for
selection.**

### 7.9 The pre-registered 11.5 metric

`artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json` (a rerun is recorded
in `phase12_rerun_verification.json`, same values):

| Field | Value |
|---|---|
| metric | `pre_registered_11.5` |
| split | `test` |
| n scored | **4,000** |
| head sha256 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` |
| **accuracy** | **0.931** |
| **macro-F1** | **0.434161** |
| loss | 0.254592 |
| num classes | 19 |
| classes present | 14 (`[0,2,3,4,5,6,7,8,9,10,12,13,17,18]`) |
| classes absent | 5 (`[1,11,14,15,16]`) |
| macro-F1 denominator | **"all 19 classes (absent classes contribute 0.0)"** |
| `is_deciding_statistic` | **false** |

> ### **Accuracy 0.931 is never quoted without macro-F1 0.434161, and the ruling is `OPEN`.**
>
> The macro-F1 denominator is all 19 classes, so the five absent classes contribute `0.0` by
> construction — which is why the two numbers answer different questions. Classes 5 and 6 are *present*
> in the scored split and still score `0.0`
> (`docs/PHASE12_115_METRIC_COMPUTED.md`). The metric JSON's own `advisory` says it "reports ONE head's
> held-out accuracy and macro-F1. It selects no head, ranks nothing and compares no arms. Whether this
> constitutes a Phase 12 pass is the owner's ruling."

### 7.10 The `PLUMBING_ONLY` label — R-08

Every run record in **both** arms carries
`result_status = "PLUMBING_ONLY — fixture/loop evidence, NOT a result; the pre-registered 11.5 metric is
not computed"` (`training/fusion/train.py:121-124`). The label is **stale** for the real 20,000-sample
runs, and it was **deliberately not changed mid-experiment**: editing the constant would have left
Arm A's already-written records inconsistent with Arm B's. Both arms carry the identical text, so the
comparison is unaffected. It is flagged for an owner ruling (**R-08**) and no historical record was
rewritten (`docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §7.7). The `run_record.json` also carries
`pre_registered_metric_computed: false` and a matching `final_val.note`.

---

## 8. Change-VQA head — external GPU, cached change features

This is the artifact with the most complete external-training contract in the repository. Its
authoritative guide is `docs/R02_KAGGLE_TRAINING_GUIDE.md` (37,543 B, 19 points); the short form is
`RUNBOOK_CHANGE_VQA_KAGGLE.md`; the pre-flight checklist is `PRE_KAGGLE_READINESS_REPORT.md` (33 items).

### 8.1 What it is

A **two-stage** head over cached change features
(`training/change_vqa/model.py`; `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §2):

```
frozen STANet detector  ->  change feature (1045-d)
                        ->  STAGE 1  class-wise change estimate (13 outputs)
                        ->  STAGE 2  question-conditioned 19-way answer
                        ->  answer + evidence + provenance
```

Stage 1's outputs are **concatenated into** stage 2's input, so the answer is computed *from* the change
estimate. That is what makes `largest_change` / `smallest_change` answerable at all: a single pooled
vector has no per-class structure to compare. It is **one trainable module with one loss**, not two
models. `ARCHITECTURE_VERSION = "change_vqa_head_v1"`.

| Component | Value | Source |
|---|---|---|
| architecture | `change_vqa_head_v1` | `artifacts/change_vqa/run/model_metadata.json` |
| parameters | **1,453,912** | `run_record.json` → `model.parameters` |
| `trunk_dim` | 512 | `model.trunk_dim` |
| `text_dim` | 256 | `model.text_dim` |
| dropout | 0.10 | `model.dropout` |
| stage-1 estimator | `Linear(1045, 256) → GELU → Linear(256, 13)` | `training/change_vqa/model.py` |
| answer head | `Linear(fused, 512) → GELU → Dropout → Linear(512, 256) → GELU → Linear(256, 19)` | same |

The architecture was chosen from four measured facts
(`R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §2): the answer space is **closed and has 19 members**
(rules out a generative decoder); answers are short (`yes`, `buildings`, `10_to_20`); `label1`/`label2`
ship for all 2,968 scenes (class-wise change is **computable supervision**, not a guessed target); and
the budget is ≤ 3 h on T4×2 (rules out training anything large).

### 8.2 The frozen dependency

The head reads features produced by the **frozen** LEVIR STANet change detector
(`artifacts/change/levir_change_v001/head.pt`, sha256
`c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa`, 63,231,009 B, 15,779,969 params).
The detector is never updated. A missing checkpoint is a **hard error** unless
`--allow-untrained-detector` is passed — "because a cache built from an untrained detector still trains,
still evaluates, and still reports a number" (`RUNBOOK_CHANGE_VQA_KAGGLE.md` §3). The notebook verifies
the checkpoint by size and sha256 and **refuses to continue** on a mismatch; this was verified by
injecting a 23-byte file in its place.

### 8.3 The feature caches

Training consumes two caches, both identified by a **spec hash**:

| Cache | Spec | Hash |
|---|---|---|
| change features (1045-d) | `change_feat_v1` | `change_cache_spec` **`c801326f85a185f8`** (trained; untrained spec is `714efac5b0e6a6d1`) |
| question features (384-d) | MiniLM-L6-v2 | `text_cache_spec` **`d2801ea1a314354a`** |

(`artifacts/change_vqa/run/model_metadata.json`.) The feature extractor re-runs the detector's
submodules and **trains nothing**: `training/change_vqa/features.py` → `ChangeFeatureExtractor`
("Trains no"). `FEATURE_SPEC = "change_feat_v1"`, `DEFAULT_IMAGE_SIZE = 256`,
`CHANGE_FEATURE_DIM = 1045` = 1024 level_pooled + 5 change_stats + 16 change_grid. The trainer refuses a
cache built under a different spec, and the specialist returns an empty answer with `degraded=True` on a
spec mismatch rather than answering from the wrong feature space.

**Cost:** feature extraction is the bottleneck, not head training. Measured **≈ 0.25 s/scene on CPU**
(64 scenes in 22 s including startup); ≈ 8 min for 2,000 scenes. Head training is ≈ 0.5 s/epoch over 26
batches at the smoke scale. The T4 figure is **not measured and not claimed**.

### 8.4 The training procedure

The run record's `argv` is the exact invocation
(`artifacts/change_vqa/run/run_record.json` → `argv`):

```
--prepared-dir /kaggle/working/outputs/change_vqa/prepared
--output-dir   /kaggle/working/outputs/change_vqa/run
--data-root    /kaggle/input/datasets/creatorballs/cdvqa-dataset
--device       cuda
--time-limit-seconds 10800
--epochs       40
--batch-size   256
--seed         42
--patience     6
```

The trainer's own defaults are `epochs=40`, `batch_size=128`, `seed=42`, `patience=6`,
`time_limit=10800 s` (`training/change_vqa/train.py`), with `DEFAULT_LR = 1e-3`,
`DEFAULT_WEIGHT_DECAY = 1e-4`, `DEFAULT_WARMUP_RATIO = 0.05`, `DEFAULT_GRAD_CLIP = 1.0`,
`DEFAULT_MIN_DELTA = 1e-4`. The guide instructs: *"Do not change these to get past an error."*

The procedure, in order:

1. **Assemble the code** as a Kaggle dataset (249 files, 248 define the digest; the guide's §1a holds the
   digest value and the recompute snippet).
2. **Assemble the CDVQA data** as a second Kaggle dataset (§4 of the readiness report: `annotations/`
   53 MB + `im1/` 1,128 MB + `im2/` 1,156 MB + `label1/` 22 MB + `label2/` 22 MB ≈ 2.4 GB). A **split
   integrity gate** refuses to continue unless the four split counts match exactly.
3. **Attach the frozen STANet checkpoint.**
4. **Create the notebook with T4 ×2 and Internet on** (MiniLM weights download).
5. **Discover paths** (marker search, not an assumed mount slug) and **verify the STANet digest**.
6. **Run the discovery, environment and integrity cells** — `integrity clean : True`, no size errors.
7. **Prepare Train and Val only** (targets → change features → question features). Held-out splits are
   prepared later, *after* the head is fitted, so no held-out label is read before the model is frozen.
8. **Train** (time-budgeted at 3 h; the budget is checked before every batch).
9. **Evaluate** on Test and Test2, masked and unmasked, with a guard over all three artifacts and a
   check that reads `eval_summary.json` back.
10. **Return the export directory** — the unit of review.

### 8.5 Optimisation

`artifacts/change_vqa/run/run_record.json` → `optimization`:

| Field | Value |
|---|---|
| optimizer | **AdamW** |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| scheduler | `cosine_with_warmup` |
| grad clip | 1.0 |
| AMP | **fp16** (`torch.float16`) with **GradScaler** |
| `amp_reason` | `cuda autocast (float16) with GradScaler` |
| `native_bfloat16` | **false** |
| total steps planned | 10,320 |
| warmup steps | 516 |
| epochs requested | 40 |
| **epochs completed** | **14** |

Loss weights (`optimization.loss_weights`, and `training/change_vqa/model.py`):

| Term | Weight |
|---|---|
| answer (CE) | 1.0 |
| delta (MSE) | 1.0 |
| magnitude (BCE) | 1.0 |
| total_changed (BCE) | **0.5** |

### 8.6 Selection

`run_record.json` → `selection`:

| Field | Value |
|---|---|
| metric | **Val answer accuracy** |
| **best epoch** | **8** |
| **best accuracy** | **0.700018** |
| patience | 6 |
| `min_delta` | 0.0001 |
| **`stop_reason`** | **`early_stopping`** |

The trainer's `FORBIDDEN_SPLITS` are `("Test", "Test2")` and the training split is `Train` with `Val`
as the selection split (`training/change_vqa/train.py`). `model_metadata.json` records
`test_splits_used: false`, `detector_trained: true`, `seed: 42`, `epoch: 8`.

### 8.7 The state vocabulary — `TRAINED_UNVERIFIED`

`run_record.json` → `state` is **`"TRAINED_UNVERIFIED"`**, with the note:

> *"training produces an artifact, not a verified capability. R-02 reaches VERIFIED only after the
> returned checkpoint has been evaluated on the held-out split."*

`confidence.method` is `"uncalibrated"` — raw softmax plus a top1−top2 margin; **nothing is fitted**.
The post-training state constant is `POST_TRAINING_STATE = "TRAINED_UNVERIFIED"`
(`training/change_vqa/train.py`).

### 8.8 The byte-identity promotion gate

The returned checkpoint was **promoted** through a byte-identity gate. `artifacts/change_vqa/run/
PROMOTION.json` (schema `change_vqa_promotion_v1`, promoted 2026-09-22):

| Property | Value | Key path |
|---|---|---|
| sha256 | `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a` | `artifact.sha256` |
| bytes | 5,822,809 | `artifact.bytes` |
| architecture | `change_vqa_head_v1` | `artifact.architecture` |
| parameters | 1,453,912 | `artifact.parameters` |
| non-finite tensors | **0** | `artifact.non_finite_tensors` |
| weights modified during promotion | **false** | `artifact.weights_modified` |
| byte-identical to source | **true** | `source.byte_identical_to_source` |
| hash agrees across | `model_metadata.json`, `run_record.json`, `hashes.json` | `source.hash_agrees_across` |
| config hash | `78f1e3700da15aa1` | — |
| epoch selected | 8 | — |
| val answer accuracy | 0.700018 | — |
| `stop_reason` | early_stopping | — |
| frozen dependency | `artifacts/change/levir_change_v001/head.pt`, sha256 `c5ef3127…e9fa` | — |
| state before | `TRAINED_UNVERIFIED` | — |
| state after | **`PROMOTED`** | — |

The promotion verification also records 93 passed / 0 failed / 0 skipped, and the two test-set scores:

| Split | accuracy | macro-F1 | n scored | baseline (global majority) |
|---|---|---|---|---|
| Test | 0.697626367 | 0.378373275 | 39,686 | 0.311546 |
| Test2 | 0.651469262 | 0.372308516 | 31,036 | 0.178728 |

with `metric_ruling: OPEN`.

> ### What the promotion gate does — and does not — establish.
>
> **It establishes** that the returned bytes are the bytes that were trained, unmodified. It is a
> **byte-identity** gate: `weights_modified: false`, `byte_identical_to_source: true`, zero non-finite
> tensors, and the digest agreeing across three independently-written records.
>
> **It does not confer `VERIFIED`.** `PROMOTION.json` records `state after: PROMOTED`, and the metric
> ruling is `OPEN`. Promotion is an *acceptance of the artifact*, not a *validation of the capability*.
> The change-VQA head is quoted under **two test sets** (Test 0.697626/0.378373 and Test2
> 0.651469/0.372309) and its ruling is `OPEN`; never quote one test set alone
> ([`BENCHMARKS.md`](BENCHMARKS.md) §4.4).

### 8.9 The two AMP/loss defects and their fixes

The external run history is instructive and is recorded rather than hidden
(`PRE_KAGGLE_READINESS_REPORT.md` §1b; `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §4):

| # | Defect | Why it blocked the run | Fix |
|---|---|---|---|
| **K1** | `F.binary_cross_entropy` on **sigmoid probabilities** under CUDA autocast | `aten::binary_cross_entropy` is registered as an outright **ERROR** under CUDA autocast → `RuntimeError: … unsafe to autocast` at step 1 on a T4 | a helper that casts both operands to fp32 **and** invokes the op with autocast disabled (casting alone is not sufficient) |
| **K3** | **Saturated-sigmoid BCE → NaN loss** | `BCE(sigmoid(z))`'s backward at `p = 1.0` is clamped to a **finite** `−9.99999996e11`, so `GradScaler` cannot flag it; under fp16 the cast node turns it into `−inf`, the sigmoid backward multiplies by `0`, and `inf × 0 = nan`. The probability form is also **wrong** before it is unstable: PyTorch clamps BCE at 100.0 per element, so a saturated element reports 65.0 where the truth is 30.0 | the loss is computed from the **pre-sigmoid logits** with `binary_cross_entropy_with_logits` — same function mathematically, exact numerically |

The K3 CPU reproduction (`PRE_KAGGLE_READINESS_REPORT.md` §1b):

| Quantity | Pre-fix | Fixed |
|---|---|---|
| parameter tensors with non-finite gradients | **4 / 20** | **0 / 20** |
| total loss at the saturated element | `133.53` (clamped) | `472.16` (exact) |
| gradient w.r.t. the logit | `nan` | `-1.0` |
| single saturated element, `z = +30, t = 0` | `100.0` forward (clamped) | `30.0` forward (exact) |

**`--no-amp` is not the fix** — it would have hidden the NaN while leaving the loss silently clamped and
wrong. The published contract is unchanged: `class_mag` is still `sigmoid(mag_raw)`, `class_delta` still
`tanh`, `total_changed` still `sigmoid`; the logits are an **addition** used only by the loss.

### 8.10 The serving wiring

`app/serving.py` wires `change_vqa` through the registry `builders=` override with **the same**
`CHANGE_CHECKPOINT` the change detector receives, plus `CHANGE_VQA_HEAD =
artifacts/change_vqa/run/head.pt`. Absent artifacts are passed as `None`, never as a fabricated path —
the builder's contract is that missing **degrades** while **corrupt** raises `ModelLoadError`.
`configs/base.yaml` is byte-identical (`88434f7f…`) and `Config.hash` is still `78f1e3700da15aa1`
(finding F2, `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §5).

### 8.11 The 3-hour budget contract

`HARD_STOP_SECONDS = 3 * 3600 = 10800` (plan §46). The budget is checked **before every batch**; on
exhaustion the run saves `checkpoint_last.pt`, flushes the log, and exits with a structured
`time_limit_reached` reason rather than dying silently. The recorded run used **64.278 s** of a
10,800 s budget (`run_record.json` → `budget`). If `epochs_completed` is 0, the time limit is shorter
than one epoch — raise it (`RUNBOOK_CHANGE_VQA_KAGGLE.md` §8).

---

## 9. VLM LoRA adapter — external GPU, PEFT

### 9.1 What it is

A PEFT LoRA adapter on the **frozen** `HuggingFaceTB/SmolVLM-500M-Instruct` (revision
`a7da5b986cb5`, Apache-2.0). It is attached to the model's **language-model projections only**; the
vision tower and the connector are frozen.

### 9.2 Hyperparameters — every value

From `configs/base.yaml` §`training`, corroborated by
`.scratch/phase6_real_adapter/phase6_adapter/run_manifest.json` → `config` and `lora`:

| Hyperparameter | Value | Source |
|---|---|---|
| base model | `HuggingFaceTB/SmolVLM-500M-Instruct` | `run_manifest.json` → `config.base_model` |
| revision | `a7da5b986cb5` | `config.revision` |
| PEFT version | **0.19.1** | `run_manifest.json` → `environment.peft`; `adapter_config.json` |
| `lora_rank` (`r`) | **16** | `config.lora_rank` |
| `lora_alpha` | **32** | `config.lora_alpha` |
| `lora_dropout` | **0.05** | `config.lora_dropout` |
| target modules | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` | `config.lora_target_modules` |
| `n_target_modules` | **224** | `lora.n_target_modules` |
| `all_trainable_in_text_model` | **true** | `lora.all_trainable_in_text_model` |
| precision | **fp16** | `config.precision` |
| batch size | 2 | `config.batch_size` |
| gradient accumulation | 8 | `config.gradient_accumulation` |
| effective batch size | **16** | `config.effective_batch_size` |
| learning rate | 0.0002 | `config.learning_rate` |
| epochs | 1 | `config.epochs` |
| weight decay | 0.01 | `config.weight_decay` |
| warmup ratio | 0.05 | `config.warmup_ratio` |
| gradient checkpointing | **true** | `config.gradient_checkpointing` |
| `max_seq_length` | 512 | `config.max_seq_length` |
| `processor_longest_edge` | **512** | `config.processor_longest_edge` |
| `do_image_splitting` | true | `config.do_image_splitting` |
| seed | 42 | `config.seed` |
| `save_every_steps` | 500 | `config.save_every_steps` |
| `max_wall_seconds` | 27,000 | `config.max_wall_seconds` |
| train / val ratio | 0.8 / 0.1 | `config.train_ratio`, `config.val_ratio` |
| `negative_ratio` | 1.0 | `config.negative_ratio` |
| `rgb_percentiles` | `[2.0, 98.0]` | `config.rgb_percentiles` |

### 9.3 The target-module regex — and the vision-tower hazard

`training/vlm/lora.py` defines:

```
LANGUAGE_MODEL_TARGET_REGEX = r"^model\.text_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
```

The anchoring is load-bearing. The module docstring records the hazard: the **vision tower's** attention
projections are *also* named `q_proj`/`k_proj`/`v_proj`, so a plain module-name **list** (or an
unanchored regex) would silently attach LoRA deltas to the vision tower as well. The anchored regex
scopes the adapter to `model.text_model.*`. The run manifest confirms the outcome:
`all_trainable_in_text_model: true`, `observed_prefixes: ["base_model.model."]`, and all 224 target
module names begin `model.text_model.layers.<n>.…` (`run_manifest.json` → `lora.target_module_names`).
`training/vlm/config.py` also carries `LORA_TARGET_MODULES` (the seven names) as the declarative list.

### 9.4 fp16 because a T4 is SM 7.5 — a recorded plan deviation

The master plan's §43 specifies **bf16** for this stage. The run uses **fp16**, and the deviation is
recorded so it is auditable — `run_manifest.json` → `config.plan_deviations`:

> *"precision: plan section 43 specifies bf16; fp16 is used because T4 is SM 7.5 and has no bf16 tensor
> cores (finding C-6). Recorded so the deviation is auditable."*

This is finding **C-6**, frozen in `docs/ARCHITECTURE_FREEZE.md` §4 (`training.precision: fp16`,
reason "T4 = SM 7.5, no bf16 tensor cores"). `training/vlm/trainer.py` **refuses** bf16 outright.

### 9.5 The corpus

The VLM instruction pairs come from a **BigEarthNet-S2 single-label** subset
(`run_manifest.json` → `config.corpus_root`:
`/kaggle/input/datasets/creatorballs/bigearth-net-s2-single-label/BigEarthNet-S2`), **not** the reBEN v2
fusion corpus. This matters and is stated explicitly.

| Property | Value | Key path |
|---|---|---|
| instruction families | `["presence"]` | `corpus.families` |
| n samples (questions) | 49,464 | `corpus.n_samples` |
| n requested patches | 28,000 | `corpus.coverage.n_requested` |
| n matched | **24,732** | `corpus.coverage.n_matched` |
| **coverage fraction** | **0.883286** | `corpus.coverage.coverage_fraction` |
| n unmatched | **3,268** | `corpus.coverage.n_unmatched` |
| n single-label | 24,732 | `corpus.coverage.n_single_label` |
| **single-label fraction** | **1.0** | `corpus.coverage.single_label_fraction` |
| n multi-label | **0** | `corpus.coverage.n_multi_label` |
| scene blocks (train / val / test) | 1,160 / 482 / 202 | `corpus.scene_counts` |
| patches (train / val / test) | 17,471 / 3,375 / 3,886 | `corpus.split_info.patches_by_split` |
| samples (train / val / test) | 34,942 / 6,750 / 7,772 | `corpus.split_counts` |
| scene key | `ben_<tile>:<k>` | `corpus.split_info.scene_key` |
| render | RGB `[B04, B03, B02]`, per-band percentile stretch 2/98, uint8 | `corpus.render` |

The corpus carries **two warnings verbatim** (`corpus.warnings`):

> *"3268 of 28000 patches have no row in the manifest and carry NO labels; they are excluded rather than
> guessed (88.3% coverage)"*
>
> *"every one of the 24732 matched patches is single-label; the BigEarthNet corpus at large averages
> ~2.95 labels per patch (max 11), so this subset cannot support multi-label enumeration questions"*

The split policy is `release_partition_keyed_by_T2_blocks` with a leakage check calling
`evaluation.leakage.assert_no_scene_overlap` (`corpus.split_info`). The block reconstruction is
conservative by construction: *"blocks are reconstructed from this corpus's patches, not the full
release; that can only split a block further, never merge across a partition."* The render note records
that the stretch is **per-patch, not corpus-global**, so inference needs no training-set statistics.

### 9.6 Training state

`run_manifest.json` → `training_state`:

| Field | Value |
|---|---|
| `epochs_completed` | **1** |
| `best_val_loss` | 0.10925133040291257 |
| checkpoints | `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000` |
| `learning_rates` | a warmup ramp from ~3.67e-6 upward |

The evaluation budget (`evaluation_budget`) is 1,000 per split, drawn once and reused for baseline and
adapted (contract item U): `selected_per_split: {test: 1000, val: 1000}`,
`truncated: {adapted_test: false, adapted_val: false, baseline_test: false, baseline_val: false}`.

### 9.7 Metrics and the acceptance decision

`run_manifest.json` → `metrics` (adapted, val, n = 1000): `exact_match: 0.911`, `f1: 0.911794`,
`precision: 0.907298`, `recall: 0.916335`, confusion `tp 460 / fp 47 / tn 451 / fn 42`. The baseline
(`baseline`, n = 1000): `exact_match: 0.491`, `f1: 0.099115`, confusion `tp 28 / fp 35 / tn 463 / fn 474`.

The **decision** (`run_manifest.json` → `decision`) is **`REJECTED`**:

- V1 passed: *"val baseline=49.10 pp, adapted=91.10 pp, delta=+42.00 pp; required (V1) >= +5.00 pp"*.
- V2 failed: *"V2 failed: 3 class(es) dropped more than 1.0 pp on validation"* — the three classes are
  `Mixed forest` (−6.4516 pp), `Transitional woodland, shrub` (−9.375 pp) and `Agro-forestry areas`
  (−4.6512 pp), each listed in `decision.class_failures`.

> **`USABLE ≠ ACCEPTED`.** The adapter's metrics are usable — a later test adjudication records
> `exact_match 0.963`, `f1 0.96432`, baseline test 0.468 (`artifacts/vlm/run1_test_recovery/
> test_adjudication.json`) — but the artifact is **`ACCEPTANCE-REJECTED`** for promotion, and the
> deployed caption/VQA path uses the **unadapted** model. The rejection is preserved in the record
> (`artifacts/vlm/phase6_closure.json`), not scrubbed. See [`MODELS.md`](MODELS.md) §3.6 and
> [`EVALUATION.md`](EVALUATION.md) §4.5.

### 9.8 Finding F5-2 — the processor cost

The processor's default `longest_edge` is 2048, which upscales 512-px tiles 4× and then splits them into
**17 sub-images** (`pixel_values (1, 17, 3, 512, 512)`, 1,142 prompt tokens). Pinning
`processor_longest_edge: 512` yields `pixel_values (1, 1, 3, 512, 512)`. The plan estimated a **4×** cost
overrun; the **measured** figure is **~17×**. This is finding **F5-2**, frozen in
`docs/ARCHITECTURE_FREEZE.md` §4 ("Tiles must not be upscaled or split").

### 9.9 Finding F5-3 — the image token

SmolVLM requires one `<image>` token per image in the prompt; hand-written prompt strings raise
`ValueError`. Prompts are therefore always built through `processor.apply_chat_template()`. This is
finding **F5-3** and is enforced in `specialists/vqa/prompts.py`.

### 9.10 The adapter digests

| Item | Value | Source |
|---|---|---|
| adapter dir tree hash | `5c6b86317d1e65962702dc9e377009b3df41cc13de1b15bceccb70ad977775e7` | `run_manifest.json` → `adapter_sha256`; `artifacts/vlm/run1_test_recovery/adapter_verification.json` |
| `adapter_model.safetensors` sha256 | `07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e` | `.scratch/phase6_real_adapter/phase6_adapter/ARTIFACT_SHA256SUMS.json` |
| manifest check | `clean: true`, 14 files in manifest / 14 on disk, `mismatched_files: []` | same |
| `checkpoint-1500` | `7273588e…` | `ARTIFACT_SHA256SUMS.json` |
| `checkpoint-2000` | `bf249943…` | `ARTIFACT_SHA256SUMS.json` |

> **The promoted adapter is the end-of-training top-level save, not `checkpoint-2000`.** Three distinct
> digests exist; conflating them is a real error. See [`MODELS.md`](MODELS.md) §1.2 and §3.6.

| Component | Parameters | Source |
|---|---|---|
| trainable (LoRA) | **8,683,520** (1.682312 % of total) | `run_manifest.json` → `trainable_params`, `trainable_fraction` |
| frozen `model.text_model` | 361,944,000 | `frozen_params` |
| frozen `model.vision_model` | 86,433,024 | `frozen_params` |
| frozen `model.connector` | 11,796,480 | `frozen_params` |
| frozen `other` | 47,308,800 | `frozen_params` |
| frozen total | 507,482,304 | sum of the four above |
| base model total | 516,165,824 | `docs/MODELS.md` §2.2 |

---

## 10. Calibration — fitted, not trained

Calibration is a **fitted** post-hoc step, not a trained artifact, and it is included here because it is
the one place a temperature is set.

`artifacts/calibration_v001.json` (schema `calibration_v1`):

| Field | Value | Key path |
|---|---|---|
| specialist | `change_vqa` | `specialist` |
| **temperature** | **0.9772731820958189** | `temperature_scaling.temperature` |
| fitted on | `Val`, n = 16,441 | `provenance.fitted_on`, `provenance.n_samples` |
| checkpoint sha256 | `cfae5e43…` | `provenance.checkpoint_sha256` |
| **ECE before** | **0.013755** | `metrics.ece_before` |
| **ECE after** | **0.014929** | `metrics.ece_after` |
| improvement | **−0.001174** | — |
| NLL before / after | 0.689741 / 0.689631 | — |
| `type_mask_applied` | false | — |

> ### **The temperature made calibration WORSE.**
>
> ECE went **0.013755 → 0.014929 — worse**. It is retained only because it is in the frozen config, and
> this is a **measured negative result**, not an improvement. `docs/` records it as such
> ([`EVALUATION.md`](EVALUATION.md) §4.7, [`BENCHMARKS.md`](BENCHMARKS.md) §4.7). Nothing else is fitted:
> the change-VQA head ships `confidence.method: "uncalibrated"`.

---

## 11. The reproducibility contract for training

1. **Seed 42** everywhere (`project.seed`), recorded in every run record's
   `reproducibility.seed`. The change-VQA record additionally documents
   `torch_seeded: true`, `numpy_seeded: true`, `python_random_seeded: true`,
   `torch_cuda_manual_seed_all: true`, `cudnn_deterministic: true`, `cudnn_benchmark: false`, and
   `deterministic_algorithms_enabled: false` with its reason (*"not enabled: it raises on operators
   lacking a deterministic implementation, which would abort the run rather than make it
   reproducible"*).
2. **Precision `fp16`** on CUDA, never bf16 (§2.2). CPU runs are fp32 with autocast a no-op.
3. **Every artifact records the frozen config hash** `78f1e3700da15aa1`; a config edit moves the hash
   and invalidates the artifact (§1.3).
4. **`save_every_steps: 500`.** Checkpoints are archived as provenance, **not released** as model
   weights ([`MODELS.md`](MODELS.md) §1.2).
5. **Training guides state their own entry status** and **never** claim a trained artifact is a verified
   capability (`docs/R02_KAGGLE_TRAINING_GUIDE.md`).
6. **Splits are by scene or group**, never by sample (§4.4; [`DATASETS.md`](DATASETS.md) §7).
7. **The public test split is immutable** and off-limits to training code; hidden data is never accessed
   (`evaluation.immutable_public_test: true`, `evaluation.hidden_data_access: false`).
8. **No one-command retrain for the external artifacts.** The repository ships the *contract*, the
   *promotion gate*, the *evaluation path* and the *serving wiring* for them — not a retraining harness
   (`docs/REPRODUCIBILITY.md` §8.5).

---

## 12. What was NOT trained — exhaustive

| Item | State | Evidence |
|---|---|---|
| **Backbone fine-tuning (any)** | **NOT DONE** — all backbones frozen | `docs/ARCHITECTURE_FREEZE.md` §2, §5 |
| **Router on the test split** | **NOT RUN** — `n_test_examples_scored: 0`, `test_split_touched: false` | `artifacts/router/threshold_sweep_val.json` |
| **Any end-to-end / joint training** | **NOT RUN** — no system-level training exists | `docs/ARCHITECTURE_FREEZE.md` §5 |
| **Change head at a second resolution** | **NOT RUN** — 448 was REJECTED for grounding, not retrained for change | §5.7 |
| **Change head at a longer cosine schedule (T_max = 60)** | **NOT RUN** — P2 in `docs/CHANGE_TRAINING_EXPERIMENT_PLAN.md`, "PREPARED, NOT EXECUTED" | same |
| **Change-head threshold retune** | **CLOSED / eliminated** — the lever is 0.37× the epoch noise | `docs/PHASE9_FREEZE.md` §5 |
| **Benchmark adapters (trained)** | **NOT RUN** — adapters are evaluation code, not trained | `evaluation/benchmark_adapters/` |
| **Arm B (optical-SAR) training beyond the five seeds** | **NOT RUN** — the comparison concluded; Arm A retained | `docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §7 |
| **The §4-registered optical-SAR arm B** | **NON-CONSTRUCTIBLE** — never built, never run | same, §2 (D-01) |
| **Arm C (optical-SAR)** | **NOT INTRODUCED** — the arm set is frozen at two | same, §2 (D-01) |
| **Change-VQA head retrained after promotion** | **NOT RUN** — one external run; the returned checkpoint was promoted, not retrained | `artifacts/change_vqa/run/PROMOTION.json` |
| **VLM adapter promoted** | **REJECTED** — trained, but `ACCEPTANCE-REJECTED`; deployed path uses the unadapted model | `artifacts/vlm/phase6_closure.json`; `run_manifest.json` → `decision.status: REJECTED` |
| **VLM adapter on a second instruction family** | **NOT RUN** — `instruction_families: ["presence"]` only | `run_manifest.json` → `corpus.families` |
| **Any multi-label VLM training** | **NOT RUN** — the subset is 100 % single-label | `run_manifest.json` → `corpus.warnings` |
| **Calibration re-fit after the negative result** | **NOT RUN** — the temperature is retained only because it is in the frozen config | §10 |
| **Router retrain with the test split scored** | **NOT RUN** | §4.7 |
| **Grounding head at 448** | **REJECTED by measurement**, not retrained | §5.7 |
| **BigEarthNet multi-label (reBEN) fusion training** | **NOT RUN** — the local subset is single-label; metrics are not comparable to published numbers | [`DATASETS.md`](DATASETS.md) §6.5 |
| **Cross-dataset generalisation training** | **NOT RUN** | [`DATASETS.md`](DATASETS.md) §8 |
| **Distributed / multi-node training** | **NOT DONE** — modular monolith, one process | `docs/ARCHITECTURE_FREEZE.md` §5 |
| **Kubernetes / queues / microservices around training** | **NOT DONE** | `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §6 ("No overbuild") |

### 12.1 Things that trained but whose *quality* is not established

| Artifact | Trained? | Quality established? |
|---|---|---|
| `change` head | yes (external GPU) | **yes** — the only `VERIFIED` headline (pooled IoU 0.8122) |
| `change_vqa` head | yes (external GPU) | metrics measured; **ruling `OPEN`**; promotion is byte-identity only |
| `optical_sar` head | yes (local CPU, 10 runs) | measured (accuracy 0.931 / macro-F1 0.434161); **ruling `OPEN`** |
| `router` adapter | yes (local CPU) | val-only, ungated, n = 86; **test NOT RUN** |
| `grounding` head | yes (local CPU) | **did not beat the zero-shot baseline** on val IoU (0.0946 < 0.0972) |
| `vlm` LoRA adapter | yes (external GPU) | metrics usable; **`ACCEPTANCE-REJECTED`** |

---

## 13. What is NOT RUN / OPEN / BLOCKED for this topic

**NOT RUN**

- The router **test split** (`n_test_examples_scored: 0`).
- Any **end-to-end / joint training**; no system-level training exists.
- The change head's **P2** longer-cosine variant (`docs/CHANGE_TRAINING_EXPERIMENT_PLAN.md`,
  "PREPARED, NOT EXECUTED").
- The change head at a **second resolution**.
- Any **retrain** of the external artifacts after their runs.
- **Multi-label** BigEarthNet training (local subset is single-label).
- **Cross-dataset generalisation** training.

**OPEN**

- The **change-VQA** metric ruling (`PROMOTION.json` → `metric_ruling: OPEN`).
- The **optical-SAR** ruling (`pre_registered_115_metric.json` → `is_deciding_statistic: false`;
  the artifact's own `advisory` says the ruling is the owner's).
- **R-03** — calibration: nothing is fitted beyond the (negative-result) temperature; the change-VQA
  head ships `method: "uncalibrated"`.
- **R-08** — the stale `PLUMBING_ONLY` `result_status` label on the optical-SAR run records.
- The **change registration gate** (`docs/PHASE9_FREEZE.md` §7) — 58.7 % false-positive rate on the
  serving path.
- **R-01** — the BigEarthNet → SmolVLM LoRA adaptation gap (the adapter was rejected).
- **F5** — the A–R test-area lettering is a reconstruction (documentation caveat;
  `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md` §8).

**BLOCKED**

- Nothing blocks the external training runs. `PRE_KAGGLE_READINESS_REPORT.md` §11 lists six unresolved
  items, and explicitly states: *"None block the external run."*
- The one *recorded* external blocker that **was** crossed: the optical-SAR Arm-B feature cache did not
  exist and required a ~2.7 h CPU extraction (`docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md` §3).
  It was completed (`armB_seed{100..104}` run records exist) and the comparison concluded.

---

## 14. Where the evidence lives

| Topic | Artifact(s) |
|---|---|
| Router training + sweep | `artifacts/router/router_adapter_v001/metadata.json`, `artifacts/router/threshold_sweep_val.json`, `router/adapter.py`, `router/dataset.py`, `docs/PHASE4_ROUTER_REPORT.md` |
| Grounding training | `artifacts/grounding/remoteclip_grounding_v001/{run_record,training_metadata}.json`, `artifacts/grounding/remoteclip_grounding_v001/eval_result_{canonical,matched6}.json`, `scripts/train_grounding.py`, `training/grounding/`, `docs/PHASE7_RESOLUTION_DECISION.md`, `docs/PHASE8_GROUNDING_HEAD_DECISION.md`, `docs/PHASE8_HANDOFF.md` |
| Change training | `artifacts/change/levir_change_v001/{run_record,training_metadata,model_metadata}.json`, `artifacts/change/eval_test/eval_result.json`, `artifacts/change/levir_real_data_verification.json`, `artifacts/change/threshold_sweep_val.json`, `docs/PHASE9_GPU_HANDOFF.md`, `docs/PHASE9_GPU_RUN_RESULTS.md`, `docs/PHASE9_REAL_DATA_VERIFICATION.md`, `docs/PHASE9_FREEZE.md`, `docs/CHANGE_TRAINING_EXPERIMENT_PLAN.md` |
| Optical-SAR sweep + production head | `artifacts/optical_sar/fusion_head_v001/arm{A,B}_seed{100..104}/`, `arm{A,B}_seed_variance_report.json`, `artifacts/optical_sar/fusion_head_production_v001/{production_head_record,pre_registered_115_metric,phase12_rerun_verification}.json`, `training/fusion/train.py`, `docs/PHASE14_OPTICAL_SAR_DECISIONS.md`, `docs/PHASE14_GATE_F_DECISION_RECORD_2026-09-20.md`, `docs/PHASE12_115_METRIC_COMPUTED.md`, `docs/PHASE12_LABEL_POLICY_DECISION.md` |
| Change-VQA external run | `artifacts/change_vqa/run/{PROMOTION,run_record,model_metadata,hashes}.json`, `docs/R02_KAGGLE_TRAINING_GUIDE.md`, `RUNBOOK_CHANGE_VQA_KAGGLE.md`, `R02_CHANGE_VQA_IMPLEMENTATION_STATUS.md`, `PRE_KAGGLE_READINESS_REPORT.md`, `training/change_vqa/` |
| VLM LoRA | `.scratch/phase6_real_adapter/phase6_adapter/{run_manifest.json,adapter_config.json,ARTIFACT_SHA256SUMS.json}`, `artifacts/vlm/run1_test_recovery/`, `artifacts/vlm/phase6_closure.json`, `RUNBOOK_PHASE6_VLM_KAGGLE.md`, `training/vlm/` |
| Calibration | `artifacts/calibration_v001.json` |
| Frozen configuration | `configs/base.yaml`, `docs/ARCHITECTURE_FREEZE.md`, `docs/PHASE9_FREEZE.md`, `docs/REPRODUCIBILITY.md` §2 |
| Leakage / firewall | `evaluation/leakage.py`, `evaluation/manifest_freeze.json` — see [`DATASETS.md`](DATASETS.md) §7 |

**Sibling documents:** [`MODELS.md`](MODELS.md) (released artifacts), [`EVALUATION.md`](EVALUATION.md)
(protocols and honesty rules), [`BENCHMARKS.md`](BENCHMARKS.md) (the numbers),
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (how to reproduce), [`DATASETS.md`](DATASETS.md) (the
corpora), [`LIMITATIONS.md`](LIMITATIONS.md) (the honest counterweight).
