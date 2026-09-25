# Performance — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` ·
`ATTEMPTED` · `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

**The single most important rule in this document: do not fabricate.** Every timing, size, budget and
constant below is copied from a file that was read, and the file is named. Where a fact is not
established, this document writes
`UNKNOWN — not established from the available evidence` rather than estimating.

---

## The three sentences you must read before anything else

1. **The system is CPU-first.** There is no GPU in the live path. Every model placement is `.to(device)`
   with `device` resolved to `"cpu"` by default; nothing calls `.cuda()`.
2. **Every timing in this document is a COMPONENT measurement, not an end-to-end latency.** A number
   such as *2.205 ms/image* is the cost of one decode step over cached features — it is not "how long a
   query takes". **There is NO end-to-end latency benchmark, no throughput measurement, no p99, no
   memory profile and no cost-per-request accounting anywhere in this repository.** That is stated
   plainly in §9 and repeated wherever a number could be misread.
3. **Cold start dominates real-world latency**, and cold start is documented as *"tens of seconds"* —
   not measured as a distribution.

> **There is no system-level performance benchmark.** The router → specialist → envelope pipeline has
> never been timed end-to-end. The only whole-system latency statement in the project is the
> **timeout budget chain** (§7), which bounds a request rather than measuring one. Where this document
> quotes a per-component figure, it names the component and the device, because a CPU figure and a T4
> figure for the same operation differ by a factor of about five (§4.8).

---

## Table of contents

1. [The performance model — CPU-first](#1-the-performance-model--cpu-first)
2. [Model footprint](#2-model-footprint)
3. [Measured component timings](#3-measured-component-timings)
4. [Cost traps that were measured and fixed](#4-cost-traps-that-were-measured-and-fixed)
5. [Tiling and the `top_k_tiles` budget](#5-tiling-and-the-top_k_tiles-budget)
6. [The timeout budget chain](#6-the-timeout-budget-chain)
7. [Cold start — the dominant real-world latency](#7-cold-start--the-dominant-real-world-latency)
8. [What is NOT measured — exhaustive](#8-what-is-not-measured--exhaustive)
9. [Optimization levers — implemented vs merely possible](#9-optimization-levers--implemented-vs-merely-possible)
10. [Status: `NOT RUN` / `OPEN` / `BLOCKED` / `DEFERRED` for this topic](#10-status-not-run--open--blocked--deferred-for-this-topic)
11. [Where the evidence lives](#11-where-the-evidence-lives)

---

## 1. The performance model — CPU-first

### 1.1 The thesis

SatQuery AI was designed against a plan whose deployment target was a Hugging Face ZeroGPU Space
(`configs/base.yaml` → `deployment.platform: huggingface-spaces`, `sdk: gradio`, `zerogpu: true`). The
**shipped** system does not run on ZeroGPU. It runs on CPU, behind a Render gateway, on a GitHub
Codespace (see [`DEPLOYMENT.md`](DEPLOYMENT.md) §3 and [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7).

The plan anticipated this. Plan §48 states the architectural position in one line:

> *"ZeroGPU is an accelerator. It is not our architectural dependency."*
> (`docs/MASTER_ARCHITECTURE_PLAN.md` §48)

Plan §49 adds the loading rule — *"Do not permanently place every model on GPU"* — and §50 fixes the
per-workflow model sets (§2.5 below). The implementation honoured both: no code path requires a GPU,
and the model cache holds at most one model (`cache_max_models: 1`, §2.4).

### 1.2 Device resolution

Device selection is a single property on the frozen config object (`core/config.py:86-91`):

```python
@property
def device_preference(self) -> str:
    override = os.environ.get("SATQUERY_DEVICE")
    if override:
        return override
    return "cuda" if _torch_cuda_available() else "cpu"
```

Three facts follow, and they are load-bearing for every timing in this document:

- The default is **`"cpu"`** whenever CUDA is unavailable. The measured evaluation artifacts confirm
  this was the case on the machines that produced them —
  `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` records
  `"environment": {"cuda_available": false, "requested_device": "cpu", "torch": "2.14.0+cpu"}`.
- `SATQUERY_DEVICE` overrides it, so a host with a GPU can be used without a code change.
- `core/config.py` **never** hardcodes `"cuda"`. It asks torch.

### 1.3 `.to(device)` everywhere

The deployment documentation states the placement rule explicitly:

> *"**CPU-first instead of ZeroGPU.** No code change was required — `device_preference` honours
> `SATQUERY_DEVICE` and defaults to CPU, every specialist defaults to `device="cpu"`, and all
> placement is `.to(device)` (never `.cuda()`)."* ([`DEPLOYMENT.md`](DEPLOYMENT.md) §11)

This matters for performance because it means there is **one** device knob, and the CPU figures in this
document are the figures the deployed system actually produces. The GPU figures (§4.8) come from
training/evaluation runs on a Kaggle T4 and are quoted only where a file records them.

### 1.4 There is no `torch.compile`

`deployment.torch_compile` is `false` in the frozen registry, and `core/config.py:114-119` **rejects**
`true` at load time:

```python
# --- C-8: ZeroGPU forbids torch.compile ----------------------------
if self.get("deployment.torch_compile") is True:
    errors.append(
        "deployment.torch_compile=true is forbidden: ZeroGPU does not "
        "support torch.compile (finding C-8)"
    )
```

Finding **C-8** is that ZeroGPU does not support `torch.compile`. The consequence for performance is
that **no graph compilation is used**, so there is no compile-time warm-up to amortise and no compiled
kernel speed-up. Any performance claim in this document is therefore an eager-mode figure.

### 1.5 The AMP / precision policy

Training precision is `fp16` (`configs/base.yaml` → `training.precision: fp16`), fixed by finding
**C-6**: the target accelerator is a T4 (SM 7.5), which supports `fp16` but not `bf16`. `core/config.py:107-112`
validates the value against `{"fp16", "bf16", "fp32"}`. This is a *training* setting; the released
artifacts are consumed in their stored dtype.

### 1.6 The one model-manager policy: `cache_max_models: 1`

`configs/base.yaml` declares:

```yaml
deployment:
  lazy_load: true
  cache_max_models: 1
```

The consequence is stated in `app/space_app.py:118-121`:

> *"The controller is built once and cached: `cache_max_models: 1` means at most one model is resident
> anyway, so rebuilding per request would thrash the cache and pay a cold start every time."*

And `app/space_app.py:230-235` ties the asset-store sizing to the same policy:

> *"the Space is where `inspect_raster` reads them and where `cache_max_models: 1` serializes their
> consumption."*

So the deployed process holds **one model at a time**. A request that needs a different specialist than
the last one pays an unload-then-load cost — this is the mechanism behind the *"Sequential-request test
under `cache_max_models=1`"* row in `docs/DEPLOYMENT_ARCHITECTURE.md` §7, which is recorded
**NOT DONE — environment-blocked** (it requires a reachable upstream).

### 1.7 What the plan intended for loading

Plan §49 gives the lazy-loading loop:

```
request
  ↓
determine specialists
  ↓
load only those specialists
  ↓
execute
  ↓
release unused models
```

`agent.unload_after_workflow: true` in `configs/base.yaml` encodes the final step. Whether a given
process actually releases is **`UNKNOWN — not established from the available evidence`**: the released
evaluation artifacts are batch evaluations, not serving traces, and no serving trace with `selected_models`
transitions has been captured for a multi-specialist workflow.

---

## 2. Model footprint

### 2.1 The six trained artifacts

The project publishes **six** trained artifacts — four task heads, one router adapter, one LoRA
adapter. Byte counts and digests below are from the **generated** manifest
[`../models/manifest.json`](../models/manifest.json) (`artifacts[*]`), cross-checked against
[`../models/checksums.sha256`](../models/checksums.sha256). The manifest is produced by reading the
files (`release/tools/generate_model_manifest.py`); where this document and the manifest disagree, the
manifest wins.

| # | `id` | Task | Path | **Bytes** | MiB | Kind |
|---|---|---|---:|---:|---:|---|
| 1 | `change_head` | `change` | `artifacts/change/levir_change_v001/head.pt` | **63,231,009** | 60.30 | trained head |
| 2 | `change_vqa_head` | `change_vqa` | `artifacts/change_vqa/run/head.pt` | **5,822,809** | 5.55 | trained head |
| 3 | `optical_sar_fusion_head` | `optical_sar` | `artifacts/optical_sar/fusion_head_production_v001/head.pt` | **14,427,457** | 13.76 | trained head |
| 4 | `grounding_head` | `grounding` | `artifacts/grounding/remoteclip_grounding_v001/head.pt` | **12,639,041** | 12.05 | trained head |
| 5 | `router_adapter` | `router` | `artifacts/router/router_adapter_v001/adapter.pt` | **211,961** | 0.20 | trained adapter |
| 6 | `vlm_lora_adapter` | `vlm` | `.scratch/phase6_real_adapter/phase6_adapter/adapter_model.safetensors` | **34,798,048** | 33.19 | LoRA adapter |

Every artifact carries `status: "PRESENT"` and `config_hash: "78f1e3700da15aa1"`.

### 2.2 The total, as arithmetic

The manifest does not publish a total, so the total is computed here **from the six byte counts above**,
and labelled as arithmetic rather than as a recorded figure:

```
63,231,009   (change)
 5,822,809   (change_vqa)
14,427,457   (optical_sar)
12,639,041   (grounding)
   211,961   (router)
34,798,048   (vlm LoRA)
-----------
131,130,325 bytes  (sum of the six manifest byte counts)

131,130,325 / 1,048,576 = 125.05 MiB
131,130,325 / 1,000,000 = 131.13 MB (decimal)
```

So the six released artifacts together are **131,130,325 bytes ≈ 125 MiB ≈ 131 MB**. The headline
"~125 MiB" is this sum, and it is the whole of what the project releases as weights.

> **Note the units.** The manifest stores raw bytes. This document converts to MiB (1,048,576 bytes) and
> to decimal MB (1,000,000 bytes) and always says which. A figure quoted without its unit is a figure
> that cannot be checked.

### 2.3 The frozen backbones — NOT redistributed

The six artifacts are small modules that sit on top of **frozen** backbones. No backbone is fine-tuned,
and no backbone weight is redistributed; each is fetched from the Hugging Face Hub, pinned by revision,
on first use ([`MODELS.md`](MODELS.md) §2). The sizes below are recorded in [`MODELS.md`](MODELS.md) §2
and in the comments of `configs/base.yaml`.

| Role | Repository | Revision | Size | Source |
|---|---|---|---|---|
| Router encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | **90.9 MB** | `configs/base.yaml` (`router.revision` comment); [`MODELS.md`](MODELS.md) §2 |
| VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` | **~1015 MB** safetensors | `configs/base.yaml` (`vlm.revision` comment); [`MODELS.md`](MODELS.md) §2 |
| Grounding encoder | `chendelong/RemoteCLIP` (`RemoteCLIP-ViT-B-32.pt`) | `bf1d8a3ccf2d` | **605.2 MB** | `configs/base.yaml` (`grounding.checkpoint_revision` comment); [`MODELS.md`](MODELS.md) §2 |
| Optical-SAR encoder | `antofuller/CROMA` (`CROMA_base.pt`) | `0dd28e3d633b` | **777.6 MB** (777,563,846 bytes) | `configs/base.yaml` (`croma.checkpoint_revision` comment); [`MODELS.md`](MODELS.md) §2 |
| Change encoder | torchvision `resnet18` (`IMAGENET1K_V1`) | — | size not recorded | `specialists/change/stanet.py`; [`MODELS.md`](MODELS.md) §2.5 |

**Backbone total, as arithmetic.** Summing the four quoted sizes exactly as they are written:

```
 90.9 MB   (MiniLM)
1015   MB   (SmolVLM, safetensors)
 605.2 MB   (RemoteCLIP)
 777.6 MB   (CROMA)
---------
2488.7 MB  ≈ 2.49 GB if every backbone were resident at once
```

Two caveats, both honest:

- **The units are not uniform.** SmolVLM is quoted as *"~1015 MB safetensors"*; CROMA is quoted as
  *"777.6 MB (777,563,846 bytes)"*. 777,563,846 bytes is 741.5 MiB, not 777.6 MiB, so the MB figures
  in this table are decimal MB as recorded, not MiB. The sum above is therefore a decimal-MB sum.
- **2.49 GB is not a resident footprint.** Under `cache_max_models: 1` (§1.6) at most **one** backbone
  is resident at a time. The relevant per-request footprint is the largest single backbone plus its
  head, not the sum.
- **The change encoder (ResNet-18) size is not recorded** in any file read:
  `UNKNOWN — not established from the available evidence`.

### 2.4 What is resident, per workflow

Plan §50 fixes the model set each workflow needs. This is the design statement; the implementation
loads specialists lazily (`lazy_load: true`), so on a given request only the listed models are
constructed.

| Workflow | Models the plan names (§50) |
|---|---|
| VQA / caption | MiniLM, SmolVLM |
| Grounding | MiniLM, RemoteCLIP, grounding head, SmolVLM (explanation) |
| Change | MiniLM, STANet, SmolVLM (explanation) |
| Optical-SAR | MiniLM, CROMA, fusion head, SmolVLM (explanation) |

Plan §50's conclusion is the reason the modular specialist architecture exists at all: *"This is why
the modular specialist architecture matters."* A grounding request does not pay for CROMA, and a change
request does not pay for RemoteCLIP — **but under `cache_max_models: 1`, alternating between two
workflows does pay a reload each time** (§1.6).

### 2.5 Training checkpoints are NOT released

Training intermediates exist on the training machines (for example the grounding head's
`checkpoint_last.pt`, and the VLM adapter's `checkpoint-1500/`, `checkpoint-2000/`) but are **archived
as provenance, not released as weights** ([`MODELS.md`](MODELS.md) §1.2; `docs/TRAINING.md` §8). For the
VLM adapter this is concrete: the promoted adapter is the end-of-training top-level save
(sha256 `07c76a75…`), **not** `checkpoint-1500` (`7273588e…`) and **not** `checkpoint-2000`
(`bf249943…`) — three distinct digests (`artifacts/vlm/phase6_closure.json` →
`artifact_verification.promoted_adapter`).

### 2.6 Parameter counts — what is known and what is not

The manifest records `parameters: null` for four of the six artifacts, deliberately: the generator does
not open the checkpoints ([`MODELS.md`](MODELS.md) §1.4). Parameter counts that *were* measured
elsewhere are:

| Artifact | Parameters | Source |
|---|---:|---|
| `router_adapter` | **51,725** (adapter, over a frozen 22,713,216-param encoder) | `artifacts/router/router_adapter_v001/metadata.json`; [`architecture/04-router.md`](architecture/04-router.md) §8 |
| `grounding_head` | **1,052,677** (`head_parameters`) | `artifacts/grounding/remoteclip_grounding_v001/training_metadata.json` |
| `change` (whole model) | **15,779,969** (`model_parameters`) | `artifacts/change/levir_change_v001/training_metadata.json` |
| `optical_sar_fusion_head` | **1,201,711** (`head_parameters`) | `artifacts/optical_sar/fusion_head_v001/armA_seed100/training_metadata.json` |
| `change_vqa_head` | **1,453,912** | `../models/manifest.json` (`parameters`); `artifacts/change_vqa/run/PROMOTION.json` |
| `vlm_lora_adapter` | **8,683,520** trainable (of 516,165,824 base; 507,482,304 frozen) | `artifacts/vlm/phase6_closure.json` → `artifact_verification.trainable_params` |

**The 50,822-versus-51,725 discrepancy is recorded, not smoothed.** `router/adapter.py:22`,
`router/train.py:8` and a comment in `configs/base.yaml` all say **50,822**. The measured count is
**51,725**, and the arithmetic that settles it is in [`architecture/04-router.md`](architecture/04-router.md)
§8 (768 + 49,280 + 774 + 516 + 129 + 129 + 129 = 51,725). Status: the 51,725 figure is `MEASURED`; the
50,822 figure is a **stale comment** and the discrepancy is **`OPEN`** (documentation only — no
behaviour depends on either number).

Backbone parameter counts, from [`MODELS.md`](MODELS.md) §2:

| Backbone | Parameters | Source |
|---|---:|---|
| MiniLM encoder | 22,713,216 | `configs/base.yaml` comment; `artifacts/router/threshold_sweep_val.json` → `adapter_encoder.parameters` |
| SmolVLM-500M-Instruct | 516,165,824 (base) | `artifacts/vlm/phase6_closure.json` |
| RemoteCLIP ViT-B/32 | 151,277,313 | [`MODELS.md`](MODELS.md) §2.3 (`VERIFIED_PARAMETERS`) |
| CROMA-base | 194,365,440 | [`MODELS.md`](MODELS.md) §2.4 |

---

## 3. Measured component timings

### 3.1 The inventory

**Every figure below is a component measurement on a specific device with a specific `n`. None is an
end-to-end latency.** The "Meaning" column states exactly what was timed.

| Component | Measurement | Device | `n` | Meaning | Source |
|---|---|---|---:|---|---|
| Grounding head decode (threshold, `top_k=20`) | **2.205 ms/image** | cpu | 16,159 | head decode per image, features cached | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` → `results.head_threshold.latency_ms_per_image` |
| Grounding head decode (argmax) | **0.655 ms/image** | cpu | 16,159 | head decode, argmax strategy | same file → `results.head_argmax.latency_ms_per_image` |
| Grounding eval total (threshold) | **35.6 s** | cpu | 16,159 | whole canonical eval, threshold strategy | same file → `results.head_threshold.seconds` |
| Grounding eval total (argmax) | **10.6 s** | cpu | 16,159 | whole canonical eval, argmax strategy | same file → `results.head_argmax.seconds` |
| Grounding eval total (zero-shot matched) | **17.9 s** | cpu | 16,159 | whole canonical eval, zero-shot decode | same file → `results.zero_shot_matched.seconds` |
| Grounding matched6 eval (threshold) | **2.158 ms/image, 34.9 s** | cpu | 16,159 | second protocol, threshold strategy | `artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json` → `results.head_threshold.*` |
| Grounding matched6 eval (argmax) | **0.652 ms/image, 10.5 s** | cpu | 16,159 | second protocol, argmax strategy | same file → `results.head_argmax.*` |
| Change eval total | **55.359 s** | cuda | 2,048 | whole LEVIR-CD-256 test eval | `artifacts/change/eval_test/eval_result.json` → `metrics.seconds` |
| Change eval per image | **≈ 27.03 ms/image** | cuda | 2,048 | *arithmetic*: 55.359 s / 2,048 | derived from the row above |
| Router adapter training (probe) | **0.28 s for 20 epochs** over 4,096×384 | cpu | 4,096 | training on a cached embedding matrix | `router/adapter.py:22-23`; `configs/base.yaml` (F4-2 comment) |
| Router adapter training (real run) | **4.92 s total** | cpu | 576-example corpus, 60 epochs | full run incl. cache fingerprint, 3 split evals, artifact write | `artifacts/router/router_adapter_v001/metadata.json` → `duration_seconds`; [`architecture/04-router.md`](architecture/04-router.md) §5 |
| Router encoder throughput | **0.118 s / 64 queries** | cpu | 64 | frozen MiniLM encode | `router/encoder.py:14`; [`architecture/04-router.md`](architecture/04-router.md) §5 |
| Router embedding-cache build | **2.256 s** | cpu | — | computing the corpus embedding cache | `artifacts/router/cache/embeddings_0f33f14403e21641.json` → `seconds` |
| RemoteCLIP encoder (zero-shot, 224) | **20.0 ms/image** (mean), **20.9 ms** (p90) | cuda (T4) | 16,159 | encoder latency at resolution 224 | `docs/PHASE7_RESOLUTION_DECISION.md` §"Measured detail" |
| RemoteCLIP encoder (zero-shot, 448) | **31.8 ms/image** (mean), **32.9 ms** (p90) | cuda (T4) | 16,159 | encoder latency at resolution 448 | same doc |
| RemoteCLIP encoder (224, CPU) | **~97 ms/image** | cpu | — | CPU encoder figure, quoted in the same doc | `docs/PHASE7_RESOLUTION_DECISION.md` §"What this establishes" |
| RemoteCLIP peak VRAM (224) | **592.1 MB** | cuda | 16,159 | peak VRAM during the experiment | `docs/PHASE7_RESOLUTION_DECISION.md` |
| RemoteCLIP peak VRAM (448) | **599.8 MB** | cuda | 16,159 | peak VRAM during the experiment | `docs/PHASE7_RESOLUTION_DECISION.md` |
| VLM caption generation | **11.6 s** | cpu | 1 | one 512×512 tile, pinned processor | `docs/PHASE5_VLM_CONTRACT.md` smoke test; [`architecture/05-specialists.md`](architecture/05-specialists.md) §7.8 |
| VLM VQA generation | **5.5 s** | cpu | 1 | one 512×512 tile, pinned processor | same |
| CROMA forward | **0.89 s** for a batch of 2 at 120 px | cpu | 2 | first real forward pass | [`MODELS.md`](MODELS.md) §2.4 |
| Change training run | **4012.29 s** total | cuda | 20 epochs | full training run | `artifacts/change/levir_change_v001/training_metadata.json` → `duration_seconds` |
| Grounding head training | **~37–50 s / epoch** | cpu | 20 epochs | per-epoch training time | `artifacts/grounding/remoteclip_grounding_v001/training_metadata.json` → `epochs[].seconds` |
| Optical-SAR head training | **~3.0–3.2 s / epoch** | cpu | 20 epochs | per-epoch training time | `artifacts/optical_sar/fusion_head_v001/armA_seed100/training_metadata.json` → `epochs[].seconds` |
| change_vqa training run | **64.278 s** (`elapsed_seconds`) | (run record) | — | run elapsed, `time.monotonic` clock | `artifacts/change_vqa/run/run_record.json` → `elapsed_seconds` |

### 3.2 Reading the grounding rows correctly

The grounding rows are the most likely to be misread, so they are spelled out.

- **2.205 ms/image is the HEAD DECODE, not the encoder.** `eval_result_canonical.json` records
  `"device": "cpu"` and `"grid": 7`. The 2.205 ms is the cost of the trained head turning cached 7×7
  feature grids into boxes. The **encoder** cost is separate and much larger: the same file's sibling
  experiment records **~97 ms/image on CPU** and **20.0 ms/image on a T4**
  (`docs/PHASE7_RESOLUTION_DECISION.md`). A grounding request "costs one encode plus the head" — the
  encoder dominates, and the encoder figure is the one from the resolution experiment, not from the
  canonical eval.
- **The eval totals (35.6 s, 34.9 s, 17.9 s, 10.6 s) are whole-corpus batch runs**, not per-request
  latencies. They include the decode of all 16,159 records.
- **The two protocols are separate.** Canonical and matched6 are different protocols with different
  numbers; the honesty rule is that neither is quoted alone
  (`DOCS_STYLE_GUIDE.md` §3). 2.205 ms/image belongs to canonical; 2.158 ms/image belongs to matched6.

### 3.3 Reading the change row correctly

`artifacts/change/eval_test/eval_result.json` records `"device": "cuda"` and `metrics.seconds: 55.359`
over `n: 2048`. The per-image figure **≈ 27.03 ms/image is arithmetic performed here** (55.359 / 2048),
not a recorded key — it is labelled as arithmetic wherever it appears. It is a **GPU** figure; no CPU
change-eval figure exists in any artifact read.

### 3.4 The device asymmetry is the point

The grounding eval is CPU (2.205 ms/image for the head); the change eval is CUDA (55.359 s total). These
two figures **cannot be compared** — different devices, different components, different corpora. The
only place a CPU-vs-GPU ratio for the *same* operation is recorded is RemoteCLIP: **20 ms (T4) vs ~97 ms
(CPU), about 5×** (`docs/PHASE7_RESOLUTION_DECISION.md`). That 5× is the project's only measured
device-speed ratio and should not be generalised to other components without evidence.

### 3.5 The critical caveat — component ≠ end-to-end

> **A component timing is not an end-to-end latency, and this document does not add component timings
> together to produce one.** Adding "VLM caption 11.6 s" + "RemoteCLIP encode 97 ms" + "head decode
> 2.205 ms" would produce a number that appears nowhere in any artifact and is wrong for at least three
> reasons: the components do not all run on one request; the devices differ; and construction/load cost
> (which dominates on a cold process, per `core/controller.py`'s timing semantics — see
> [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §3.6) is not in
> any of them.

---

## 4. Cost traps that were measured and fixed

This section is the performance heart of the project: costs that were **measured**, were larger than the
plan predicted, and were **fixed in code or config**.

### 4.1 F5-2 — the VLM processor overrun: the plan said 4×, the measurement said ~17×

**This is the single largest measured cost trap in the project.**

`docs/PHASE5_VLM_CONTRACT.md` records the mechanism and the magnitude. The processor's default
`longest_edge` is **2048**. A 512 px tile is therefore resized up to 2048, and then `do_image_splitting`
cuts the 2048-px image into 4×4 = 16 sub-images of 512 each, **plus 1 overview** = **17 images**.

```
INPUT: one 512×512 RGB tile

DEFAULT  (size.longest_edge = 2048, do_image_splitting = True)
    pixel_values  (1, 17, 3, 512, 512)     <- 17 images
    input_ids     (1, 1142)                <- 1142 tokens

PINNED   (size = {"longest_edge": 512})
    pixel_values  (1,  1, 3, 512, 512)     <-  1 image
```

(`docs/PHASE5_VLM_CONTRACT.md` §"The headline finding"; `configs/base.yaml` F5-2 comment;
[`architecture/05-specialists.md`](architecture/05-specialists.md) §5.3.)

The plan's finding **C-3** said the processor "would 4× upscale our 512 px tiles". The probe confirms
the *mechanism* and corrects the *magnitude*: the real figure is **~17×**, not 4×. The Phase 5 record
states why the distinction matters:

> *"The plan got the mechanism right and the magnitude wrong. Recorded because the difference between
> '4×' and '17×' is the difference between a cost you absorb and a cost that makes the deployment quota
> non-viable."* (`docs/PHASE5_VLM_CONTRACT.md`)

**The fix, and how it is enforced.** `configs/base.yaml` sets `vlm.processor_longest_edge: 512`, and
`core/config.py:121-142` **ties that pin to `image.tile_size`** so it is a control, not a comment:

```python
proc_edge = self.get("vlm.processor_longest_edge")
tile_size = self.get("image.tile_size")
if not isinstance(proc_edge, int) or proc_edge < 1:
    errors.append(...)
elif isinstance(tile_size, int) and proc_edge > tile_size:
    errors.append(
        f"vlm.processor_longest_edge={proc_edge} exceeds "
        f"image.tile_size={tile_size}; the processor would upscale every "
        f"tile and then split it into ~17 sub-images (finding F5-2)"
    )
```

A config that would re-introduce the upscale **fails to load**. The pin is confirmed in the real load
path, not just the probe: the Phase 5 smoke test reports `max_images_seen = 1`, and the record notes
*"Unpinned it would read 17."*

**The cost in one line:** 17 images and 1142 prompt tokens per call versus 1 image and ~64 tokens. At
1142 prompt tokens per call, *"an un-pinned processor would spend a user's entire daily quota on one or
two queries"* (`docs/PHASE5_VLM_CONTRACT.md` §"What this means for the other phases"). The context
budget after the pin: **~64 tokens** at 1 image / 512 px, leaving room inside a 2048-token context for
`max_new_tokens=128` plus multi-tile composites.

### 4.2 F4-1 — the tokenizer ceiling, and a deliberate halving

Finding **F4-1**: the MiniLM tokenizer's own ceiling is **256** (verified by probe). The config sets
`router.max_length: 128` — a deliberate truncation **well inside** the ceiling, *"not the model limit.
Satellite queries are short; halving the sequence halves attention cost for no measurable accuracy
loss."* (`configs/base.yaml`). `router/encoder.py` **asserts** the configured value is ≤ 256, because
*"truncating above the ceiling is a silent no-op"* — a control that appears to work and does nothing.

### 4.3 F5-1, F5-3, F5-4 — the loader and prompt traps

- **F5-1.** `AutoModelForVision2Seq` **does not exist** in transformers 5.17.0 (not merely deprecated —
  referencing it raises `AttributeError`). The loader class is resolved by feature detection over
  `("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForMultimodalLM")`, never
  hardcoded (`specialists/vqa/model.py::resolve_loader_class`). Cost consequence: a hardcoded class name
  would crash at import; the fallback makes the module robust to a moving dependency.
- **F5-3.** SmolVLM requires one `<image>` token per image; hand-written prompt strings raise
  `ValueError`. Every prompt is built through `processor.apply_chat_template()`
  (`configs/base.yaml` → `vlm.prompt_must_use_chat_template: true`, enforced at `core/config.py:144-150`).
- **F5-4.** The dtype kwarg is not discoverable by signature (`from_pretrained` is `**kwargs`-only).
  Resolved by a call-time fallback over `("dtype", "torch_dtype")` — *"a call-time fallback, not an
  introspection"* (`docs/PHASE5_VLM_CONTRACT.md` §F5-4).

### 4.4 C-8 — `torch.compile` is forbidden, and enforced

Covered in §1.4. The performance consequence is that **no compilation is available**, and the config
guard prevents an operator from enabling it on a ZeroGPU target that would reject it.

### 4.5 C-7 — `image_resolution % 8 == 0`

CROMA asserts `image_resolution % 8 == 0` internally. `configs/base.yaml` sets `croma.image_resolution: 120`,
giving 15×15 = **225 patches**; `core/config.py:97-105` rejects any value that is not a multiple of 8.
The cost consequence: a non-conforming resolution would fail inside CROMA at construction rather than
producing a subtly different patch grid.

### 4.6 The change model's attention memory line — computed, not assumed

STANet's PAM builds a full `positions × positions` attention matrix. At batch 8, fp32
([`MODELS.md`](MODELS.md) §3.1; [`architecture/05-specialists.md`](architecture/05-specialists.md) §19):

```
layer1   8 * 4096^2 * 4 =  537.0 MB   <- exceeds a Kaggle session's budget
layer2   8 * 1024^2 * 4 =   33.5 MB   <- fine
layer3   8 *  256^2 * 4 =    2.1 MB   <- fine
layer4   8 *   64^2 * 4 =    0.13 MB  <- fine
```

Attention is therefore applied at **layers 2, 3 and 4** and **skipped at layer 1**, where the difference
features are fused directly. The decision is **recomputed on every forward pass** from the actual batch
size and a byte budget (`DEFAULT_ATTENTION_BUDGET_BYTES = 256 * 1024 * 1024`), *"so changing batch size
or tile size moves the line correctly rather than silently overrunning memory"*. This is a **recorded
deviation** from a literal STANet reproduction, not a silent substitution.

### 4.7 The 448 resolution — a cost that was measured and REJECTED

Raising the grounding encoder from 224 to 448 was tested at full scale (16,159 records) and **rejected**.
The measured cost and the measured (negative) benefit (`docs/PHASE7_RESOLUTION_DECISION.md`):

| metric | 224 | 448 | delta |
|---|---|---|---|
| token grid | 7×7 = 49 | 14×14 = 196 | 4.0× tokens |
| attention cost (n²) | 1× | 16× | — |
| latency mean | **20.0 ms** | 31.8 ms | **1.59×** |
| latency p90 | **20.9 ms** | 32.9 ms | 1.57× |
| peak VRAM | **592.1 MB** | 599.8 MB | +7.7 MB |
| wall time | **~8.5 min** | ~11.2 min | 1.32× |
| mean best IoU | **0.0972** | 0.0825 | **−0.0147** |
| Recall@0.10 | **0.3298** | 0.2599 | −0.0699 |

**448 is worse on every quality metric and slower.** The paired test agrees: `mean paired diff −0.0147`,
`95% CI [−0.0160, −0.0134]`, `t = −22.63`, `n = 16159`. The config comment in `configs/base.yaml`
records the same verdict and notes 448 was better on 8.5% of records and worse on 20.9%. The resolution
is frozen: `grounding.image_size: 224`, `grounding.resolution_frozen: true`.

> **A note on the early probe.** An *earlier, small-`n`* resolution probe
> (`artifacts/grounding/resolution_experiment.json`, `n_samples: 12`) reported a latency ratio of
> **1.77** and a verdict of `INCONCLUSIVE` (the pre-registered rule requires ≥ 30 samples). The
> **1.59×** figure above is the full-scale paired result. The two are different measurements; this
> document quotes the full-scale one as the decision and records the small-`n` one as a probe that was
> correctly never treated as evidence.

### 4.8 F5-5 — the cost of a wasted generation, and the gate that prevents it

The most important Phase 5 finding is not a latency trap but a **correctness** trap with a cost
dimension: fed a 512×512 array of uniform random bytes, the VLM produced a fluent, fabricated scene
description (`docs/PHASE5_VLM_CONTRACT.md` §F5-5). A prompt cannot fix this — *"a 500M-parameter VLM
will generate something for any input tensor it is handed"*. The fix is a **deterministic input gate
upstream of the model** (`preprocessing/quality.py`), which classifies input as `STRUCTURED`, `FLAT`,
`NOISE`, `TOO_SMALL`, or `INVALID_VALUES` and **blocks the VLM call** for the blocking verdicts
(`NOISE`, `INVALID_VALUES`). The performance benefit is direct: a noise or non-finite tile is rejected
**before** a generation is spent. The measured gate behaviour (autocorrelation carries it):

| input | verdict | autocorrelation | entropy |
|---|---|---|---|
| structured ramp+texture | `structured` | 0.9524 | 7.5810 |
| uniform random noise | `noise` | −0.0082 | 7.9882 |
| salt-and-pepper | `noise` | −0.0098 | 1.0000 |
| constant / all-zero tile | `flat` | 1.0000 | 0.0000 |

This is a cost control as much as a correctness control: **the cheapest generation is the one not run.**

---

## 5. Tiling and the `top_k_tiles` budget

### 5.1 The declared policy

`configs/base.yaml` fixes the input geometry:

```yaml
image:
  max_pixels: 25000000
  tile_size: 512
  tile_overlap: 128
  max_tiles: 64
  # plan section 9.1 tile policy: whole-image thumbnail first, then top-K tiles.
  # max_tiles is the hard ceiling on tiles *examined*; top_k_tiles is how many
  # are actually sent through a specialist.
  top_k_tiles: 4
```

The two-tier budget is stated in the comment and is the whole design:

- **`max_tiles: 64`** is the hard ceiling on tiles **examined**.
- **`top_k_tiles: 4`** is how many tiles are **actually sent through a specialist**.
- **`tile_size: 512` / `tile_overlap: 128`** fix the tile geometry (a 512 px tile with 128 px overlap
  steps 384 px).
- **`max_pixels: 25000000`** caps the input image at 25 megapixels.

`core/config.py:214-216` enforces the ordering — `top_k_tiles` may not exceed `max_tiles`:

```python
# --- tiling policy -------------------------------------------------
if self.get("image.top_k_tiles", 0) > self.get("image.max_tiles", 0):
    errors.append("image.top_k_tiles cannot exceed image.max_tiles")
```

Plan §9.1 gives the policy in prose: *"whole-image thumbnail first, then top-K tiles."*

### 5.2 Why this is a cost budget, not a detail

`top_k_tiles: 4` is the project's principal **bounded-work** control on the VLM path. Each tile sent
through the specialist costs one VLM call; a 4-tile composite costs four. Combined with the F5-2 pin
(§4.1) — 1 image per call, ~64 prompt tokens — the tiling budget bounds the VLM work per query to **at
most `top_k_tiles` generations**, independent of how large the input image is (up to `max_pixels`).

### 5.3 The honest gap: the tiling *executor* was not found in the source read

This is stated plainly because it affects every tiling claim:

- The **policy** is declared in `configs/base.yaml` and **validated** in `core/config.py`.
- The **module purpose** is named in `preprocessing/__init__.py`: *"SatQuery AI preprocessing: raster
  loading, optical/SAR normalisation, tiling."*
- The **stage is referenced** in docstrings — `preprocessing/imagery.py:15` (*"Size changes belong in
  the tiling stage, which records what it did"*) and `preprocessing/quality.py:288` (*"Filling nodata
  from the raster profile belongs in the tiling work"*).
- **However**, a search for a tiling implementation (`def ...tile`, `_tiles`, `Tiler`, `tiling`) across
  `core/`, `preprocessing/`, `specialists/` and `geospatial/` returned **no implementing function** —
  only the config key, the config validation, and the docstring references above. There is no
  `preprocessing/tiling.py`.

**Status:** the tiling **budget** is `IMPLEMENTED` (config + validation); the tiling **executor** is
**`NOT FOUND` in the source read** — `UNKNOWN — not established from the available evidence` whether a
tiling stage runs in the deployed path. A reader must not assume `top_k_tiles: 4` is enforced at
runtime merely because it is validated at load time. This gap is recorded here rather than smoothed,
per `DOCS_STYLE_GUIDE.md` §1.6.

---

## 6. The timeout budget chain

The only whole-request latency statement in the project is a set of **bounds**, not a measurement. The
bounds are read from the live health payload and the deployed source
([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §2.4).

### 6.1 The three reported timeouts

| Field | Live value | Layer | What it bounds |
|---|---:|---|---|
| `config.tunnel_timeout_s` | **150.0 s** | orchestrator, tunnel path | how long a parked request waits for an agent to pick it up and return |
| `config.wake_timeout_s` | **120.0 s** | orchestrator, forward path | how long the wake loop polls `/v1/health` before giving up |
| `config.upstream_timeout_s` | **90.0 s** | orchestrator, HTTP client | the per-request timeout on the proxied call to the Codespace |

Each is an environment variable read at call time, not a constant:

```python
def _tunnel_timeout_s() -> float:
    return float(os.environ.get("SATQUERY_TUNNEL_TIMEOUT_S", "150"))

def _wake_timeout_s() -> float:
    return float(os.environ.get("SATQUERY_WAKE_TIMEOUT_S", "120"))

def _upstream_timeout_s() -> float:
    return float(os.environ.get("SATQUERY_UPSTREAM_TIMEOUT_S", "90"))
```

The tunnel timeout's own reasoning is documented: *"Generous: a cold Codespace can take ~90s to boot,
and the agent only starts polling once the model server is ready."*

### 6.2 Two further budgets not in the payload

| Budget | Value | Layer | Source |
|---|---:|---|---|
| `_WAKE_HEALTH_TIMEOUT_S` | 10.0 s | per-**poll** health timeout during a wake | `apply-test/main.py:107`, used at `:421` |
| `_WAKE_POLL_INTERVAL_S` | 2.0 s | sleep between wake polls | `apply-test/main.py:106`, used at `:437` |
| `DEFAULT_BUDGET_SECONDS` | **120.0 s** | **controller** run budget, checked *between* plan steps | `core/controller.py:121-122` |

The controller's own run budget is a **fourth** timeout in a different layer. The controller states its
honest limit: *"Budget is checked BETWEEN steps (§5.2). A true per-step timeout needs a worker process or
a signal handler, both of which conflict with the single-process monolith constraint more than they
benefit."* (`core/controller.py:548-552`).

### 6.3 The gateway's timeout bounds are derived, not chosen

`gateway/policy.py::GatewayConfig.__post_init__` refuses an `upstream_timeout_s` that is **≤ 45 s** or
**≥ 120 s** (`gateway/policy.py:259-271`):

- **≤ 45 s** — *"Kills a legitimate `grounding`/`optical_sar` call and reports it as upstream failure."*
- **≥ 120 s** — *"Outlives the Space's own budget, turning an upstream timeout into a client hang."*

The bounds are derived from the frozen config: `agent.timeout_seconds = 120` and the longest
`gpu_duration_*` = 45 s (`docs/PHASE19_FINAL_HARDENING.md` §3.2). The live value is **90.0 s**.

### 6.4 B-07 — the measured worst case ≈ 249 s

This is the one **measured** end-to-end-ish timing in the deployment story, and it is a *failure* timing.

`SATQUERY_TRANSPORT=auto` means *try the tunnel; on timeout, fall through to the forward path*. The
forward path to a **private** repo returns `302` quickly — but the wake step still consumes
`SATQUERY_WAKE_TIMEOUT_S` (120 s) **first**. So a worst-case failed request takes roughly:

```
150 s (tunnel timeout)  +  120 s (wake timeout on a 302)  ≈  249 s
```

([`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.3.)

> *"B-07 root shape: in `auto` mode a tunnel timeout **falls through** to the forward path
> (`main.py:546`), burning `wake_timeout_s = 120` on a `302` (~249 s ≈ 150 + 120)."*

**Status: B-07 is `OPEN`.** A patch (`fix-b07-forward-unavailable.patch`) was authored and verified
(`py_compile` clean, applies cleanly to the deployed `main.py`), adding distinct `forward_unavailable`
(503) and `upstream_timeout` (504) codes plus the `codespace_name` `.strip()` fix. **The patch is
prepared but NOT deployed.** The live health payload still shows the trailing `\n`, which is itself the
witness that the patch is not deployed.

### 6.5 The signature an operator reads

A request that **hangs for ≈249 s and then returns `504`** is the tunnel-gap signature, not a broken
model. The decision tree ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.3):

```
tunnel_timeout_s (150) + wake_timeout_s (120) = 270 s  (nominal)
measured                                            ≈ 249 s
```

The three signals to read, in order: (1) `tunnel.agent_connected` on a **fresh** `/api/health`; (2) the
elapsed time (≈249 s is the B-07 signature); (3) the machine code (`tunnel_offline`, `wake_timeout`,
`upstream_timeout`, `forward_unavailable`, …).

---

## 7. Cold start — the dominant real-world latency

### 7.1 The shape

The cold start is **blocking and documented**, not hidden. The orchestrator's module docstring states it:

> *"If the Codespace is stopped, this service starts it via the GitHub Codespaces REST API, polls its
> `/v1/health` until it answers 200, then proxies the request. The client simply **waits** (blocking,
> wake-then-proxy) and receives the result; the frontend independently shows 'Waking inference
> engine...' on slow responses."*

The wake loop polls `GET {base}/v1/health` every 2 s, each with a 10 s timeout, bounded by
`wake_timeout_s = 120` ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.2).

| Phase | What happens | Bound |
|---|---|---|
| 0 | the client sends `POST /api/infer` and **waits** | — |
| 1 | `GET` the Codespace via the GitHub API; `POST .../start` if not `available` | one API round trip |
| 2 | poll `GET {base}/v1/health` every 2 s, each with a 10 s timeout | ≤ `wake_timeout_s = 120` |
| 3 | on the first `200`, proxy the original request | ≤ `upstream_timeout_s = 90` |
| 4 | the response carries `X-SatQuery-State: waking` | — |

### 7.2 The honest number

Cold start is **"tens of seconds"** ([`DEPLOYMENT.md`](DEPLOYMENT.md) §8, §5). The bounded worst case is
`wake_timeout_s = 120` before a `504`.

> **Do not quote a single cold-start number.** No measured cold-start distribution exists:
> `UNKNOWN — not established from the available evidence`
> ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.2).

### 7.3 Why it dominates

Three things can be cold, and each has a different warm-up
([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.1):

| What is cold | How it warms | How long |
|---|---|---|
| the Render orchestrator | first request to any `/api/*` route | Render free tier sleep/wake cycle |
| the Codespace | `ensure_codespace_up()` starts it and polls | bounded by `wake_timeout_s = 120` |
| the HF model cache | `warm_cache.py`, or the first model-touching request | a download per model |

A cold Codespace can take **~90 s to boot** (the tunnel timeout's own reasoning), and the model cache
must then be populated. Against a per-component VLM generation of 5.5–11.6 s (§3), the cold-start path
is **the dominant term by an order of magnitude** — which is why the honest performance story is
"cold start dominates", not "the models are fast".

---

## 8. What is NOT measured — exhaustive

This is the section a reader must not skip. **Everything below is a capability the project does not
have.**

| Not measured | Status | Note |
|---|---|---|
| **End-to-end latency** | **NOT RUN — none exists** | No request has been timed router → specialist → envelope. [`LIMITATIONS.md`](LIMITATIONS.md) §3 item 17; `DOCS_STYLE_GUIDE.md` §3 |
| **Throughput / QPS** | **NOT RUN** | No load test exists; "a deployed system-level load test" is `NOT RUN` ([`DEPLOYMENT.md`](DEPLOYMENT.md) §13) |
| **p50 / p95 / p99 latency** | **NOT RUN** | No latency histogram is produced or aggregated ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §6) |
| **Memory profiling** | **NOT RUN** | The only VRAM figures recorded are the RemoteCLIP resolution experiment's peak (592.1 MB / 599.8 MB, §4.7). No CPU RSS profile, no per-model memory profile |
| **Cost-per-request accounting** | **NOT RUN** | `GPU_DURATIONS` declares intended ZeroGPU reservations but *"nothing reads `GPU_DURATIONS` back out to compute, record or report consumption"* ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §6.1) |
| **GPU latency for the VLM** | **UNKNOWN** | The only VLM figures are CPU (11.6 s caption / 5.5 s VQA); *"A GPU latency figure for this specialist is `UNKNOWN — not established from the available evidence`"* ([`architecture/05-specialists.md`](architecture/05-specialists.md) §7.8) |
| **Cold-start distribution** | **UNKNOWN** | "tens of seconds" only (§7.2) |
| **Resident-memory under `cache_max_models: 1`** | **UNKNOWN** | The sequential-request test is `NOT DONE — environment-blocked` (`docs/DEPLOYMENT_ARCHITECTURE.md` §7) |
| **A CPU change-eval latency** | **UNKNOWN** | The only change eval is on CUDA (§3.3) |
| **Encoder CPU figure for CROMA / SmolVLM** | **UNKNOWN** | Only the CROMA forward (0.89 s, batch 2) is recorded; no per-image figure |
| **APM / distributed tracing / metrics endpoint** | **absent** | None exists ([`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §6) |
| **SLO / SLA** | **absent** | None defined |

**And the two things this document deliberately does not do:**

- It does **not** sum component timings into an end-to-end figure (§3.5).
- It does **not** promote any component figure to a system-level latency claim. A component figure is
  a component figure.

---

## 9. Optimization levers — implemented vs merely possible

Honesty requires separating what is **in the code and config** from what is **a plausible idea that was
never implemented or measured**. Nothing in the right column is a claim.

| Lever | Status | Evidence / reasoning |
|---|---|---|
| **CPU-first, GPU-optional** | **IMPLEMENTED** | `device_preference` + `.to(device)`; no `.cuda()` anywhere ([`DEPLOYMENT.md`](DEPLOYMENT.md) §11) |
| **Lazy loading (`lazy_load: true`)** | **IMPLEMENTED** | `configs/base.yaml`; plan §49 |
| **Single-model cache (`cache_max_models: 1`)** | **IMPLEMENTED** | `configs/base.yaml`; bounds resident memory but serialises workflows (§1.6) |
| **Frozen backbones + small trained heads** | **IMPLEMENTED** | The whole 125 MiB artifact set is small *because* backbones are frozen ([`MODELS.md`](MODELS.md) §1) |
| **Cached-embedding router training (F4-2)** | **IMPLEMENTED + MEASURED** | 0.28 s / 20 epochs; 4.92 s for the real run (§3) |
| **VLM processor pin (F5-2)** | **IMPLEMENTED + MEASURED** | The single largest fixed cost: ~17× → 1× (§4.1) |
| **Deterministic input gate before the VLM (F5-5)** | **IMPLEMENTED + MEASURED** | Blocks a generation on noise/non-finite input (§4.8) |
| **MiniLM truncation to 128 (F4-1)** | **IMPLEMENTED** | Halves sequence/attention cost inside the 256 ceiling (§4.2) |
| **Change attention budget recomputed per forward** | **IMPLEMENTED** | Skips layer-1 attention to fit memory (§4.6) |
| **Grounding at 224, not 448** | **IMPLEMENTED (REJECTED 448)** | 448 was slower *and* worse (§4.7) |
| **Prompt built via chat template (F5-3)** | **IMPLEMENTED** | Correctness, with a cost side-effect (no retry loop) |
| **`torch.compile`** | **REJECTED** | Forbidden on ZeroGPU (C-8); enforced in `core/config.py` (§1.4) |
| **Batching tiles for the VLM** | **NOT IMPLEMENTED / NOT MEASURED** | `vlm.max_images_per_call: 1` fixes one image per call; no batched path exists. A multi-tile composite is documented as *possible* in the caption path but not measured (`docs/PHASE5_VLM_CONTRACT.md` §"Context budget") |
| **Quantisation / distillation of the backbones** | **NOT IMPLEMENTED / NOT MEASURED** | No quantised weights, no distilled model, in any file read |
| **A persistent model cache across requests** | **NOT IMPLEMENTED** | `cache_max_models: 1` holds one model; alternating workflows reload (§1.6) |
| **Warm-keeping the Codespace** | **NOT IMPLEMENTED** | Cold start is accepted and documented (§7) |
| **A per-step timeout** | **NOT IMPLEMENTED** | Budget is checked between steps only; the controller states the reason (§6.2) |
| **A latency histogram / metrics endpoint** | **NOT IMPLEMENTED** | None exists (§8) |
| **A tiling executor** | **NOT FOUND** | Policy is validated; no executor was found in the source read (§5.3) |

> **Read the right-hand column as a to-do list of ideas, not as a feature list.** None of them is
> implemented, and none of them has a measured benefit in this repository.

---

## 10. Status: `NOT RUN` / `OPEN` / `BLOCKED` / `DEFERRED` for this topic

| Item | State |
|---|---|
| An end-to-end latency benchmark | **NOT RUN** — none exists |
| A throughput / load test | **NOT RUN** |
| A latency histogram (p50/p95/p99) | **NOT RUN** |
| Memory profiling (CPU RSS, per-model) | **NOT RUN** |
| Cost-per-request accounting | **NOT RUN** |
| A measured cold-start distribution | **NOT RUN** — "tens of seconds" only |
| GPU latency for the VLM | **UNKNOWN** |
| A CPU change-eval latency | **UNKNOWN** |
| The sequential-request test under `cache_max_models=1` | **NOT DONE — environment-blocked** |
| A tiling executor in the deployed path | **NOT FOUND in the source read** |
| B-07 tunnel fallthrough (worst case ≈249 s) | **OPEN** — patch prepared, NOT deployed |
| B-02 `codespace_name` trailing `\n` | **OPEN** (cosmetic) |
| The `torch.compile` path | **REJECTED** (C-8) |
| The 448 grounding resolution | **REJECTED** (measured slower and worse) |
| The ZeroGPU/Gradio deployment target | **REJECTED** (superseded; frozen paperwork only) |
| The 50,822 vs 51,725 parameter discrepancy | **OPEN** (documentation only) |

---

## 11. Where the evidence lives

| Topic | Evidence |
|---|---|
| All sizes and budgets | `configs/base.yaml` (and the frozen `configs/deploy.yaml`, which is inert — `docs/DEPLOYMENT_ARCHITECTURE.md` §7) |
| The F5-2 processor overrun | `docs/PHASE5_VLM_CONTRACT.md`; enforced at `core/config.py:121-142` |
| Grounding component timings | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`, `.../eval_result_matched6.json` |
| The 224-vs-448 decision and encoder latency/VRAM | `docs/PHASE7_RESOLUTION_DECISION.md`; probe at `artifacts/grounding/resolution_experiment.json` |
| Change eval timing | `artifacts/change/eval_test/eval_result.json` |
| Router timings and parameter count | `router/adapter.py`; `router/train.py`; `artifacts/router/router_adapter_v001/metadata.json`; [`architecture/04-router.md`](architecture/04-router.md) §5, §8 |
| VLM CPU latency and the input gate | `docs/PHASE5_VLM_CONTRACT.md`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §7.8, §8 |
| CROMA forward | [`MODELS.md`](MODELS.md) §2.4 |
| Six artifact byte counts and digests | [`../models/manifest.json`](../models/manifest.json); [`../models/checksums.sha256`](../models/checksums.sha256) |
| Training durations | `artifacts/*/training_metadata.json`; `artifacts/change_vqa/run/run_record.json` |
| Timeouts, cold start, B-07 | [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §2.4, §7; [`DEPLOYMENT.md`](DEPLOYMENT.md) §8 |
| Gateway timeout bounds | `gateway/policy.py`; `docs/PHASE19_FINAL_HARDENING.md` §3.2 |
| What is not measured | [`LIMITATIONS.md`](LIMITATIONS.md) §3, §4; this document §8 |

### Cross-references

- [`BENCHMARKS.md`](BENCHMARKS.md) — what *quality* was measured (accuracy, IoU, F1), with the same
  component-not-system caveat.
- [`EVALUATION.md`](EVALUATION.md) — the protocols and the honesty rules behind every metric.
- [`architecture/05-specialists.md`](architecture/05-specialists.md) — the per-specialist cost traps in
  full.
- [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) — the timing
  semantics, the timeout chain, and the cold-start runbook.
- [`MODELS.md`](MODELS.md) — the artifact and backbone reference.
- [`LIMITATIONS.md`](LIMITATIONS.md) — the exhaustive gap catalogue.
- [`GLOSSARY.md`](GLOSSARY.md) — definitions of every term used here.
