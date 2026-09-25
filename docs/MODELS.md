# Models — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED`
· `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

SatQuery AI trains **six** artifacts. Four are task heads, one is a router adapter, one is a LoRA
adapter. Every backbone is **frozen** and publicly pinned by revision in `configs/base.yaml` — the
project trains small modules on top of frozen encoders, not end-to-end networks. No backbone is
fine-tuned; no backbone weight is redistributed.

> **This file is the human-readable companion to the machine-generated
> [`../models/manifest.json`](../models/manifest.json) and
> [`../models/checksums.sha256`](../models/checksums.sha256).**
> Where the two disagree, **the generated manifest wins** — it is computed by reading the files
> (`release/tools/generate_model_manifest.py`), this document is written by hand.

**The single most important rule in this document: do not fabricate.** Every number below is copied
from a file that was read, and every number names the file and (where the source is a JSON artifact)
the exact key path. Where a fact is not established, this document says so explicitly rather than
estimating.

---

## Table of contents

1. [The six trained artifacts](#1-the-six-trained-artifacts)
2. [Frozen backbones — pinned, never retrained](#2-frozen-backbones--pinned-never-retrained)
3. [Per-artifact deep reference](#3-per-artifact-deep-reference)
   - 3.0 [Enforced configuration invariants, with arithmetic](#30-enforced-configuration-invariants-with-arithmetic)
   - 3.1 [`change` — STANet-style Siamese change detector](#31-change--stanet-style-siamese-change-detector)
   - 3.2 [`change_vqa` — change question answering head](#32-change_vqa--change-question-answering-head)
   - 3.3 [`optical_sar` — CROMA-base fusion head](#33-optical_sar--croma-base-fusion-head)
   - 3.4 [`grounding` — RemoteCLIP grounding head](#34-grounding--remoteclip-grounding-head)
   - 3.5 [`router` — intent adapter over frozen MiniLM](#35-router--intent-adapter-over-frozen-minilm)
   - 3.6 [`vlm` — SmolVLM LoRA adapter (USABLE_VERIFIED, ACCEPTANCE-REJECTED)](#36-vlm--smolvlm-lora-adapter-usable_verified-acceptance-rejected)
4. [Rejected, deferred and open model decisions](#4-rejected-deferred-and-open-model-decisions)
5. [Calibration — a measured negative result](#5-calibration--a-measured-negative-result)
6. [Distribution and licensing](#6-distribution-and-licensing)
7. [Status summary — what is NOT established](#7-status-summary--what-is-not-established)
8. [Evidence index](#8-evidence-index)

---

## 1. The six trained artifacts

All six are published on the Hugging Face Hub under `thundercode/SatQuery`, one directory per task.
The table below is reproduced from [`../models/manifest.json`](../models/manifest.json)
(`artifacts[*]`), cross-checked against [`../models/checksums.sha256`](../models/checksums.sha256).

| # | `id` | Task | Artifact path | Bytes | sha256 (full) | Kind | Backbone (frozen) |
|---|---|---|---|---|---|---|---|
| 1 | `change_head` | `change` | `artifacts/change/levir_change_v001/head.pt` | 63,231,009 | `c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa` | trained head | none external — ResNet-18 (torchvision ImageNet) + PAM, trained in-project |
| 2 | `change_vqa_head` | `change_vqa` | `artifacts/change_vqa/run/head.pt` | 5,822,809 | `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a` | trained head | STANet change detector (frozen, backing the head's change features) |
| 3 | `optical_sar_fusion_head` | `optical_sar` | `artifacts/optical_sar/fusion_head_production_v001/head.pt` | 14,427,457 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` | trained head (production) | `antofuller/CROMA` (`CROMA_base.pt`, rev `0dd28e3d633b`) |
| 4 | `grounding_head` | `grounding` | `artifacts/grounding/remoteclip_grounding_v001/head.pt` | 12,639,041 | `93432f7034be91a8ffd9c1a84e3eeec00bed7832c043fe7f83d2be230284c6bb` | trained head | `chendelong/RemoteCLIP` (`RemoteCLIP-ViT-B-32.pt`, rev `bf1d8a3ccf2d`) |
| 5 | `router_adapter` | `router` | `artifacts/router/router_adapter_v001/adapter.pt` | 211,961 | `8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c` | trained adapter | `sentence-transformers/all-MiniLM-L6-v2` (rev `1110a243fdf4`) |
| 6 | `vlm_lora_adapter` | `vlm` | `.scratch/phase6_real_adapter/phase6_adapter/adapter_model.safetensors` | 34,798,048 | `07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e` | **LoRA adapter** (PEFT) | `HuggingFaceTB/SmolVLM-500M-Instruct` (rev `a7da5b986cb5`) |

Every artifact in the manifest carries `status: "PRESENT"` and `config_hash: "78f1e3700da15aa1"`.

### 1.1 What each `kind` means

| `kind` | Meaning | How it is loaded |
|---|---|---|
| `trained_head` | A module trained in-project on top of a frozen encoder. The encoder is fetched separately. | `torch.load(...)` of a `state_dict`, or the module's own `load_*` function. |
| `trained_adapter` | A small classifier attached to a frozen sentence encoder whose embeddings are cached. | `IntentAdapter.from_config_dict` + `load_state_dict`. |
| `lora_adapter` | A PEFT LoRA delta attached to a frozen VLM at load time. | `peft.PeftModel.from_pretrained(model, dir)` (`specialists/vqa/model.py::_attach_adapter`). |

### 1.2 Training checkpoints are NOT released artifacts

Training intermediates exist on the machines that trained — e.g. the grounding head's
`checkpoint_last.pt` and the VLM adapter's `checkpoint-1500/`, `checkpoint-2000/`. These are
**archived as provenance, not released as model weights** (`docs/TRAINING.md` §8). For the VLM
adapter this matters concretely: the promoted adapter is the end-of-training **top-level** save, and
it is **not** `checkpoint-2000` (three distinct digests; see §3.6).

### 1.3 Two cross-checks where the computed hash agrees with an independently-recorded value

The manifest is generated by hashing the files on disk. For two artifacts, those computed digests can
be compared against values that were recorded *independently*, at a different time, by a different
process — which is a genuine external cross-check rather than a self-consistency claim.

**Cross-check A — `change_vqa` = `cfae5e43…`.** The manifest's `sha256` for `change_vqa/head.pt`
equals, exactly:

- `artifacts/change_vqa/run/PROMOTION.json` → `artifact.sha256`;
- the same file → `source.checkpoint_sha256_in_run_record` (the digest recorded in the Kaggle run
  record, *before* promotion);
- `artifacts/calibration_v001.json` → `provenance.checkpoint_sha256` (recorded when the temperature
  was fitted, a separate step).

`PROMOTION.json` → `source.hash_agrees_across` additionally records that the digest agrees across
`model_metadata.json`, `run_record.json` and `hashes.json (run.checkpoint_sha256)`. The bytes were
copied byte-identically (`source.byte_identical_to_source: true`) and
`artifact.weights_modified: false`.

**Cross-check B — `vlm` = `07c76a75…`.** The manifest's `sha256` for
`vlm/adapter_model.safetensors` equals, exactly:

- `artifacts/vlm/phase6_closure.json` → `production_adapter`… (via
  `artifacts/vlm/run1_test_recovery/adapter_verification.json`) →
  `weights_file_sha256` = `07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e`;
- the same file → `promoted_adapter.sha256`;
- the adapter's own `ARTIFACT_SHA256SUMS.json`, against which the directory was verified
  (`manifest_check.clean: true`, `n_files_in_manifest: 14`, `n_files_on_disk: 14`,
  `mismatched_files: []`).

Both cross-checks are recorded in [`../MODEL_CARD.md`](../MODEL_CARD.md) §1 as well.

### 1.4 Why `parameters` is `null` for most artifacts

The manifest records `parameters: null` for four of the six artifacts. That is deliberate: the
generator does not open the checkpoints (opening a `.pt` to count parameters would require the model
code and torch, and would make the manifest's generation depend on the environment). Parameter counts
that *were* measured elsewhere are recorded in this document with their source. Where a parameter
count is not established, this document writes
`UNKNOWN — not established from the available evidence` rather than deriving one from file bytes.

---

## 2. Frozen backbones — pinned, never retrained

Backbones are resolved from the Hugging Face Hub on first use, pinned by revision. The revision pins
are the load-bearing part: a moving `main` would make every benchmark number unreproducible.

| Role | Repository | Revision | Size | Measured identity | Notes |
|---|---|---|---|---|---|
| Router encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | 90.9 MB | 22,713,216 params, 384-dim embeddings | tokenizer ceiling **256**; truncation set to **128** |
| VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` | ~1015 MB safetensors | 516,165,824 params (base) | processor `longest_edge` **must** be pinned (F5-2) |
| Grounding | `chendelong/RemoteCLIP` (`RemoteCLIP-ViT-B-32.pt`) | `bf1d8a3ccf2d` | 605.2 MB | 151,277,313 params; width **768**, **projected** dim **512** | patch size 32; 7×7 tokens at 224 |
| Optical-SAR | `antofuller/CROMA` (`CROMA_base.pt`) | `0dd28e3d633b` | 777.6 MB (777,563,846 bytes) | 194,365,440 params; `encoder_dim` 768 | `image_resolution` 120; asymmetric `s1_depth=6`, `s2_depth=12` |
| Change encoder | — (torchvision) | — | — | ResNet-18, `IMAGENET1K_V1` | `pretrained_used: true` in the trained artifact |

### 2.1 Router encoder — `all-MiniLM-L6-v2`

- Declared in `configs/base.yaml` under `router:` — `model`, `revision: 1110a243fdf4`,
  `max_length: 128`, `embedding_dim: 384`.
- **Finding F4-1.** The MiniLM tokenizer's own ceiling is **256** (verified by probe). `128` is a
  deliberate truncation well inside that ceiling, not the model limit. The encoder **asserts the
  configured value is ≤ 256**, because truncating above the ceiling is a silent no-op — a control that
  appears to work and does nothing.
- **Finding F4-2.** The encoder is frozen, so embeddings are cached and the adapter trains on cached
  vectors. Measured on CPU: **20 epochs over 4,096 × 384 in 0.28 s**. No GPU required.
- The revision is asserted by `router/encoder.py` (guards on revision and `max_length`).

### 2.2 VLM — `SmolVLM-500M-Instruct`

- Declared under `vlm:` — `checkpoint`, `revision: a7da5b986cb5`, `max_new_tokens: 128`,
  `do_sample: false`, `temperature: 0.0`, `processor_longest_edge: 512`, `do_image_splitting: true`,
  `max_images_per_call: 1`.
- **Finding F5-1.** `AutoModelForVision2Seq` does **not exist** in transformers 5.17.0 (it is not
  merely deprecated — referencing it raises `AttributeError`). The loader class is resolved by
  feature detection over `("AutoModelForImageTextToText", "AutoModelForVision2Seq",
  "AutoModelForMultimodalLM")`, never hardcoded (`specialists/vqa/model.py::resolve_loader_class`).
- **Finding F5-2 (cost).** The processor's default `longest_edge` is 2048, which upscales a 512-px
  tile 4× and then splits it (`do_image_splitting=True`) into **17 sub-images** and 1142 prompt
  tokens. MEASURED: default → `pixel_values (1, 17, 3, 512, 512)`; pinned → `pixel_values
  (1, 1, 3, 512, 512)`. The plan estimated a 4× cost overrun; the real figure is **~17×**. The pin
  **must** be set explicitly on the processor at construction time, and `generate` raises loudly if
  more than one image is produced for a single input.
- **Finding F5-3.** SmolVLM requires one `<image>` token per image in the prompt; hand-written prompt
  strings raise `ValueError`. All prompts are built through `processor.apply_chat_template()`.
- **Finding F5-4.** The dtype kwarg is not discoverable by signature (`from_pretrained` is
  `**kwargs`-only). Resolved by a call-time fallback over `("dtype", "torch_dtype")`.

### 2.3 Grounding — `RemoteCLIP ViT-B/32`

- Declared under `grounding:` — `checkpoint_repo`, `checkpoint_file: RemoteCLIP-ViT-B-32.pt`,
  `checkpoint_revision: bf1d8a3ccf2d`, `model_name: ViT-B-32`, `image_size: 224`,
  `resolution_frozen: true`, `nms_iou: 0.50`, `max_candidates: 20`, `confidence_threshold: 0.40`,
  `benchmark_box_scale: 100.0`, `coordinate_system: normalized_0_1`,
  `encoder_projected_dim: 512`.
- **Measured, not assumed** (`specialists/grounding/remoteclip.py`): `VERIFIED_PATCH_SIZE = 32`,
  `VERIFIED_TRANSFORMER_WIDTH = 768`, `VERIFIED_PROJECTED_DIM = 512`,
  `VERIFIED_PARAMETERS = 151_277_313`. `_verify_contract()` asserts all four against the loaded model
  at load time.
- **The 768-vs-512 distinction is the one that bites.** `visual.positional_embedding` is 768 wide and
  `visual.proj` is `(768, 512)`; the embeddings the text tower can be compared against are the
  **projected** ones (512). Using 768 anywhere here is a shape error torch would surface only at the
  similarity step — after the patch features have already been computed and cached.
- Tokens: 7×7 = 49 (+1 CLS) at 224; 14×14 = 196 (+1 CLS) at 448. `SUPPORTED_RESOLUTIONS = (224, 448)`;
  which one to *use* was an empirical question answered by the resolution experiment (§4.1), and
  neither is marked "preferred" in the encoder.

### 2.4 Optical-SAR — `CROMA-base`

- Declared under `croma:` — `checkpoint_repo`, `checkpoint_file: CROMA_base.pt`,
  `checkpoint_revision: 0dd28e3d633b`, `variant: base`, `image_resolution: 120`, `encoder_dim: 768`,
  `optical_channels: 12`, `sar_channels: 2`, `modalities: [optical, sar, joint]`,
  `modalities_used: [optical, sar, joint]`.
- **Finding C-7.** `image_resolution % 8 == 0`; native 120 → 15×15 = **225 patches**. Enforced in
  `core/config.py` and re-checked in `CROMAEncoder.__init__`.
- **Finding C-1 — CROMA is never given a mask.** The forward pass takes exactly two arguments,
  `model(SAR_images=..., optical_images=...)`. The availability mask is consumed by the **fusion
  head**, not by CROMA (see §3.3). The rationale: CROMA is a masked autoencoder, and handing it an
  availability mask invites it to reconstruct missing channels — precisely the fabrication the sensor
  adapter exists to prevent.
- First real forward pass (measured): CPU, 0.89 s for a batch of 2 at 120 px, 194,365,440 parameters,
  `n_patches = 225`, `(B, 768)` GAPs and `(B, 225, 768)` tokens confirmed. Two facts no document had
  recorded surfaced: CROMA-base is **asymmetric** (`s1_depth=6`, `s2_depth=12`) and the joint
  cross-attention is **directional** (SAR queries optical).
- **DEV-1 (constructor).** CROMA is distributed as a GitHub repository, not a pip package; the
  constructor is `use_croma.PretrainedCROMA`, and `use_croma.py` **must be vendored**. Ruled
  **ACCEPTED as an implementation detail** — no `ARCHITECTURE CHANGE` entry
  (`docs/PHASE14_OPTICAL_SAR_DECISIONS.md` §2). The vendored file is recorded at
  `specialists/optical_sar/vendor/use_croma.py`, 14,556 bytes, sha256
  `a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b`.
- **DEV-2 (input normalisation).** Resolved 2026-09-18: option (a) — match upstream's per-channel
  `mean ± 2·std` stretch immediately before the forward pass with `use_8_bit` enabled — adopted as the
  required implementation; option (c) registered as a **gated** experiment with (a) as its control;
  option (b) rejected as a standalone path. Implemented in `specialists/optical_sar/radiometry.py`
  and wired through `croma.py`.

### 2.5 Change encoder — ResNet-18, trained in-project

The change detector's encoder is **not** an external pinned backbone in the sense the others are: it
is a torchvision `resnet18`, requested with `change.pretrained: true` and recorded in the trained
artifact's embedded config as `pretrained_used: true` with `encoder: "resnet18"` and
`encoder_channels: [64, 128, 256, 512]`.

The code does **not** make pretrained weights a hard dependency. Resolution order
(`specialists/change/stanet.py`): (1) a local weights file, if given; (2) torchvision's
`IMAGENET1K_V1` download; (3) random initialisation **with an explicit warning** — because
`pretrained: true` in a config must not become a lie when a download fails. The trained artifact
records `pretrained_used: true`, so this run used ImageNet weights.

### 2.6 Backbone licensing

Backbones are **not redistributed** by this project. They are fetched from the Hub at run time and
carry their own licences (see each model's HF page). See §6.

---

## 3. Per-artifact deep reference

Each subsection below gives, for one artifact: its path, byte count, sha256, architecture, **every
hyperparameter** (from `configs/base.yaml` and the code), the backbone and its pinned revision, the
training data, the evaluation protocol, the measured numbers (with artifact key paths), the
acceptance status, and its limitations.

### 3.0 Enforced configuration invariants, with arithmetic

Two invariants are **enforced at load time**, not merely documented. Both exist because the failure
mode they prevent is a *silent shape error* — a tensor of exactly the right shape that trains to a
worse number, the hardest kind of bug to notice.

#### 3.0.1 `fusion.input_dim == 2318`

```
input_dim = len(modalities_used) * encoder_dim + optical_channels + sar_channels
          = 3 * 768              + 12             + 2
          = 2304                 + 14
          = 2318
```

`configs/base.yaml` states the arithmetic in a comment above `fusion.input_dim: 2318`, and the code
recomputes it **twice**, independently:

- `core/config.py` recomputes it as
  `len(modalities_used) * encoder_dim + optical_channels + sar_channels` at load time (finding C-1);
- `specialists/optical_sar/fusion_head.py::expected_fusion_dim` returns
  `int(modalities_used) * int(encoder_dim) + int(optical_channels) + int(sar_channels)`;
- `build_fusion_head` **raises `ModelLoadError`** if `int(input_dim) != 2318`, naming the arithmetic
  (`3*768 + 12 + 2`) in the message;
- `assemble_fusion_input` concatenates `[optical_GAP, sar_GAP, joint_GAP, optical_mask, sar_mask]`
  and **raises `SpecialistError`** if the assembled width is not the expected width.

The order of the concatenation is frozen (freeze §2.5) and **asserted rather than assumed**, because a
permutation produces a tensor of exactly the right shape that trains to a worse number.

> The mask goes **here**, not into CROMA. Freeze §2.5: channel/band dropout during fusion-head
> training is **mandatory** — it is what teaches the head to trust the availability mask. With no
> dropout the head learns to read channel 4 unconditionally, because in training channel 4 was always
> present; on a 4-band sensor channels 5–12 are *always* zero, and a head that never saw a masked
> channel treats those zeros as a measurement of blackness rather than as absence.
> `channel_dropout()` is a real function, tested directly (measured: `p=1.0` keeps 48/48; `p=0.4`
> drops to 18/48; dropped features are exactly `0.0`, not renormalised; a band the sensor never
> measured stays dropped even at `p=1.0`; the seeded RNG is deterministic).

#### 3.0.2 `grounding_head.feature_dim == 2048`

```
feature_dim = 4 * grounding.encoder_projected_dim
            = 4 * 512
            = 2048
```

The per-cell feature is

```
f_i = concat([p_i, t, p_i * t, global_pool])        -> 4 * 512 = 2048
```

where `p_i` is the patch token, `t` the text embedding broadcast to every cell, `p_i * t` the
element-wise cross-modal alignment, and `global_pool` the mean over all patches. The global term
matters: a per-cell MLP otherwise cannot see anything outside its own patch, and a 1/7-of-image
receptive field is too small for objects that span several cells.

The invariant is enforced **three** ways:

- `core/config.py` **rejects any value other than** `4 * grounding.encoder_projected_dim` at load
  time — deliberately, so the head can be validated *without importing torch*;
- `specialists/grounding/head.py::build_head` recomputes `expected = 4 * VERIFIED_PROJECTED_DIM` and
  raises `ValueError` naming both numbers if they differ;
- `GroundingHead.forward` re-checks `feat.shape[-1] != self.feature_dim` and raises, and
  `RemoteCLIPEncoder._verify_contract` asserts `visual.proj.shape[1] == 512` against the real model.

A mismatch is a silent shape error: torch raises only at the similarity step, after the patch
features are already cached.

#### 3.0.3 Other enforced invariants (summary)

| Invariant | Enforced where | Failure mode prevented |
|---|---|---|
| `croma.image_resolution % 8 == 0` | `core/config.py`, `CROMAEncoder.__init__`, `OpticalSarSpecialist.__init__` | CROMA asserts it internally (finding C-7) |
| `router.max_length <= 256` | `router/encoder.py` | truncating above the tokenizer ceiling is a silent no-op |
| `change` spatial dims divisible by 8 | `STANetStyleChangeDetector.forward` | encoder stride; a 1-px-off raster is reflect-padded and cropped back, with a warning |
| `sa_mode in {PAM, BAM}`; `BAM` raises | `STANetStyleChangeDetector.__init__` / `.forward` | a config asking for BAM fails visibly rather than silently aliasing PAM |
| `fusion_head` device must be `"cpu"` | `build_fusion_head` | a device string that would silently fall back is worse than a refusal |
| `task_dim >= 2` | `OpticalSarSpecialist.__init__` | a margin cannot exist over one class |
| exactly 2 assets for change / change_vqa / optical_sar; exactly 1 for vqa/caption | each specialist's `validate_request` | a silent "just use the first two" produces a confident answer to a question the caller did not ask |

---

### 3.1 `change` — STANet-style Siamese change detector

**Status:** `IMPLEMENTED`, `VERIFIED` — the only task whose headline metric carries the `VERIFIED`
tag, because it is the only one measured against a single, immutable public test split with a frozen
threshold.

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `change_head` | `models/manifest.json` |
| Path | `artifacts/change/levir_change_v001/head.pt` | manifest |
| Bytes | 63,231,009 | manifest |
| sha256 | `c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa` | manifest, `checksums.sha256` |
| HF path | `change/head.pt` | manifest |
| Architecture | STANet-style Siamese change detector (ResNet-18 + PAM) | manifest, code |
| `kind` | `trained_head` | manifest |
| `backbone` | `null` (trained in-project; ResNet-18 ImageNet trunk) | manifest |
| Config hash | `78f1e3700da15aa1` | manifest |
| Parameters | `UNKNOWN — not established from the available evidence` (manifest records `null`) | manifest |

#### Architecture (`specialists/change/stanet.py`)

A STANet-shaped Siamese network — shared Siamese encoder, spatial-temporal attention over feature
differences, feature-difference aggregation decoder — **reimplemented rather than vendored**, per the
Phase 0 resolution of finding C-9 (the upstream repo is Python 3.6-era and depends on `visdom`/`apex`).

```
T1 ---> [ shared encoder ] ---> f1  (4 levels)
T2 ---> [ shared encoder ] ---> f2  (4 levels)
                                  |
                      |f1 - f2| + concat([f1, f2, |f1-f2|])
                                  |
                          spatial attention (PAM)
                                  |
                        progressive decoder + skips
                                  |
                          1-channel change logit
```

- **Weights are tied, not copied.** Both branches call the **same** module instance
  (`SharedResNetEncoder`), so there is no second encoder to fall out of sync — the "shared encoder"
  requirement from plan §16 enforced by construction, not by convention.
- **Measured encoder shapes** (ResNet-18 trunk, 256×256 input):

  | Level | Channels | Grid | Positions |
  |---|---|---|---|
  | `layer1` | 64 | 64×64 | 4096 |
  | `layer2` | 128 | 32×32 | 1024 |
  | `layer3` | 256 | 16×16 | 256 |
  | `layer4` | 512 | 8×8 | 64 |

- **The attention memory line is computed, not assumed.** STANet's PAM builds a full
  `positions × positions` attention matrix. At batch 8, fp32:

  ```
  layer1   8 * 4096^2 * 4 =  537.0 MB   <- exceeds a Kaggle session's budget
  layer2   8 * 1024^2 * 4 =   33.5 MB   <- fine
  layer3   8 *  256^2 * 4 =    2.1 MB   <- fine
  layer4   8 *   64^2 * 4 =    0.13 MB  <- fine
  ```

  Attention is therefore applied at **layers 2, 3 and 4** and **skipped at layer 1**, where the
  difference features are fused directly instead. The decision is **recomputed on every forward pass**
  from the actual batch size and a byte budget (`DEFAULT_ATTENTION_BUDGET_BYTES = 256 * 1024 * 1024`),
  so changing batch size or tile size moves the line correctly rather than silently overrunning
  memory. Applied/skipped levels are returned in the output so the trace can report them
  (`ChangeOutput.to_trace`). **This is a deviation from a literal STANet reproduction, recorded as
  one** — attention at 64×64 would not fit. It is not a silent substitution.

- **Difference fusion.** `DifferenceFusion` reduces `concat([f1, f2, |f1-f2|])` (3C channels) back to
  a working width with `Conv2d(3C, out, 3, padding=1, bias=False) → BatchNorm2d → ReLU`.
- **Attention module.** `SpatialAttention` is PAM-style: 1×1 `query`/`key` convolutions to
  `hidden = max(1, channels // reduction)` with `reduction = 8`, a 1×1 `value` to `channels`, and a
  1×1 `out`; `softmax(q @ k / sqrt(hidden))` over positions, residual `x + out(attended)`.
- **Decoder.** Progressive, coarse-to-fine: `dec3` and `dec2` are `DecoderBlock(width, width, width)`;
  `dec1` is `DecoderBlock(width, width, width // 2)`; `final_up` is
  `ConvTranspose2d(width//2, width//2, 4, stride=4)`; `head` is `Conv2d(width//2, 1, 1)` with
  `FINAL_BIAS_INIT = -2.0` (LEVIR-CD's changed-pixel fraction is roughly 5–15%, so a bias of −2.0
  starts the prior near 0.12 rather than 0.5 and stops the first epochs being spent un-learning a
  saturated sigmoid).
- **Loss.** Composite BCE + Dice, `change_loss(logits, target, bce_weight=0.5, dice_weight=0.5,
  pos_weight=None)`. Dice is computed on soft probabilities so it is differentiable and directly
  optimises region overlap; `eps = 1e-6` guards the empty-target case (a tile with no change would
  otherwise be `0/0 = NaN`). `pos_weight` is left as an explicit knob rather than a hidden default,
  because Dice already addresses the class imbalance.

#### Hyperparameters — every value

From `configs/base.yaml` (`change:`):

| Key | Value |
|---|---|
| `tile_size` | 256 |
| `tile_overlap` | 0 |
| `threshold` | 0.50 |
| `min_component_pixels` | 32 |
| `encoder` | `resnet18` |
| `sa_mode` | `PAM` (`BAM` declared but not implemented; raises) |
| `pretrained` | true |
| `learning_rate` | 0.001 |
| `batch_size` | 8 |
| `bce_weight` | 0.5 |
| `dice_weight` | 0.5 |
| `levir_split.train` | 7120 |
| `levir_split.val` | 1024 |
| `levir_split.test` | 2048 |

From the trained artifact's **embedded** config
(`artifacts/change/eval_test/eval_result.json` → `checkpoint_embedded_config`), which is the config
actually baked into the weights:

| Key | Value |
|---|---|
| `width` | 128 |
| `sa_mode` | `PAM` |
| `attention_budget_bytes` | 268435456 (256 MB) |
| `encoder` | `resnet18` |
| `encoder_channels` | `[64, 128, 256, 512]` |
| `frozen_encoder` | false |
| `pretrained_used` | true |

#### Training data

LEVIR-CD-256 — the standard 256×256 change-detection benchmark. Split used (from `configs/base.yaml`):
train **7,120** / val **1,024** / test **2,048**. The test split is immutable and public.

#### Evaluation protocol

`artifacts/change/eval_test/eval_result.json`. Environment recorded in the artifact: device `cuda`,
torch `2.10.0+cu128`, Python `3.12.13`, platform `Linux-6.12.90+-x86_64-with-glibc2.35`,
`cuda_available: true`, `seconds: 55.359`. Config-hash integrity is recorded explicitly:
`config_hash: 78f1e3700da15aa1`, `checkpoint_config_hash: 78f1e3700da15aa1`,
`checkpoint_config_hash_checked: true`, `config_drift: false`,
`config_drift_acknowledged: false`.

Evaluation population: `n: 2048`, `n_images_with_change: 935`,
`mean_change_fraction: 0.0509`, `n_pixels: 134217728`, `threshold: 0.5`.

Change-fraction quantiles (`change_fraction_quantiles`): `min 0.0`, `p50 0.0`, `p90 0.197205`,
`max 0.684937`. The median tile has **zero** changed pixels, which is why pooled and macro metrics are
both reported.

#### Measured numbers

All from `artifacts/change/eval_test/eval_result.json`.

| Metric | Value | Exact key path |
|---|---|---|
| **pooled IoU** | **0.8122** | `metrics.pooled.iou` |
| **macro IoU** | **0.8457** | `metrics.macro.miou` |
| **pooled F1** | **0.8964** | `metrics.pooled.f1` |
| pooled mean IoU (`miou`) | 0.9007 | `metrics.pooled.miou` |
| pooled precision | 0.9195 | `metrics.pooled.precision` |
| pooled recall | 0.8745 | `metrics.pooled.recall` |
| macro IoU (`iou`, per-class change IoU) | 0.718 | `metrics.macro.iou` |
| macro F1 | 0.7962 | `metrics.macro.f1` |
| macro precision | 0.8506 | `metrics.macro.precision` |
| macro recall | 0.7757 | `metrics.macro.recall` |
| tp | 5,978,997 | `metrics.pooled.tp` |
| fp | 523,658 | `metrics.pooled.fp` |
| fn | 858,407 | `metrics.pooled.fn` |
| tn | 126,856,666 | `metrics.pooled.tn` |

The confusion counts are identical between the pooled and macro blocks (as they must be — they are the
same pixel population; the two blocks differ only in how the class-wise scores are aggregated).

**Pooled vs macro, and why both.** The test split is only ≈ 5 % changed pixels. Pooled IoU answers
"how well does the mask overlap overall"; macro IoU answers "how well does each class do, averaged".
Quoting one alone would hide the imbalance question.

#### Acceptance status

`VERIFIED`. This is the only task whose headline number carries the `VERIFIED` tag in
[`BENCHMARKS.md`](BENCHMARKS.md) §1, because it is the only one measured against a single, immutable
public test split with a frozen threshold.

#### Serving wiring (and why the checkpoint is not in the config)

A trained head exists and is benchmarked, yet it is **deliberately not wired into serving by default**
(`specialists/change/specialist.py`). Populating `change.checkpoint_path` in `configs/base.yaml` would
move `Config.hash`, and `scripts/eval_change.py` refuses to score on a hash drift (exit 3) — so that
one edit would invalidate the project's own benchmark number. The wiring path is therefore the
registry `builders=` override (`core/registry.py`), which injects the checkpoint without touching the
config. With no checkpoint wired, a missing artifact is a **deployment** case, not a crash: the
specialist runs the randomly-initialised model, marks the result `degraded`, and says so plainly,
because a randomly-initialised change map is a map of noise and presenting it as a detection would be
exactly the fabrication the evidence system exists to prevent. `has_checkpoint` distinguishes "a
checkpoint was loaded" from "we built a random one and are being honest about it".

#### Limitations

- **Absolute IoU is high, but this is one benchmark.** LEVIR-CD-256 is overhead optical; it says
  nothing about the hidden ISRO/SAC distribution (Cartosat-2S + RISAT).
- **Attention is skipped at layer 1** — a recorded deviation from literal STANet, forced by memory.
- **The detector is not the product path by default** (§ above).
- **Post-processing suppresses spatial claims on a mis-registered pair.** Poor co-registration drives
  confidence to 0.0 and withholds region claims; the maps are still written, but the region assertions
  are not made. This is by design (`architecture freeze §2.4`, plan §40) but it means the reported
  metric is for a well-registered benchmark, not for arbitrary pairs.
- **Minimum region size is 32 px** (`change.min_component_pixels`), so sub-32-px changes are
  discarded by construction.

---

### 3.2 `change_vqa` — change question answering head

**Status:** `IMPLEMENTED`, `MEASURED`, ruling **OPEN**.

A head that answers natural-language change questions over a temporal pair. It answers the eight CDVQA
question types (`change_or_not`, `change_ratio`, `change_ratio_types`, `change_to_what`,
`increase_or_not`, `decrease_or_not`, `largest_change`, `smallest_change`) over a **closed 19-answer
space**.

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `change_vqa_head` | `models/manifest.json` |
| Path | `artifacts/change_vqa/run/head.pt` | manifest |
| Bytes | 5,822,809 | manifest |
| sha256 | `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a` | manifest, `PROMOTION.json` |
| HF path | `change_vqa/head.pt` | manifest |
| Architecture | `change_vqa_head_v1` | `PROMOTION.json` → `artifact.architecture` |
| **Parameters** | **1,453,912** | `PROMOTION.json` → `artifact.parameters` |
| Backbone | STANet change detector (frozen, backing the head's change features) | manifest |
| Config hash | `78f1e3700da15aa1` | manifest |
| `satquery_trained` | true | `PROMOTION.json` → `artifact.satquery_trained` |
| `eval_mode` | true | `PROMOTION.json` → `artifact.eval_mode` |
| `non_finite_tensors` | **0** | `PROMOTION.json` → `artifact.non_finite_tensors` |
| `weights_modified` | **false** | `PROMOTION.json` → `artifact.weights_modified` |

#### Architecture and the frozen dependency

The head consumes **cached features**, not raw imagery: a frozen STANet detector produces a change
representation, and a frozen question text encoder (`sentence-transformers/all-MiniLM-L6-v2`) produces
a text vector; the head maps those to an answer index. Recorded feature specs
(`PROMOTION.json` → `identity`): `feature_spec: change_feat_v1`,
`change_cache_spec: c801326f85a185f8`, `text_cache_spec: d2801ea1a314354a`,
`preprocessing_version: change_vqa_preproc_v1`.

The **frozen dependency** is the STANet detector itself (`PROMOTION.json` → `frozen_dependency`):

| Field | Value |
|---|---|
| role | STANet change detector backing the head's change features |
| path | `artifacts/change/levir_change_v001/head.pt` |
| sha256 | `c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa` |
| bytes | 63,231,009 |
| `detector_trained` | true |
| `verified_byte_exact_vs_local` | true |

This digest is byte-identical to the `change` artifact in §3.1 — the head is wired to the same
checkpoint the change task releases.

**The detector is frozen and shared in spirit, not in instance** (`specialists/change/vqa_specialist.py`).
When the planner routes a change+language request it plans **both** a `change` step and a
`change_vqa` step, so the detector runs twice for one request. That is a real cost, stated rather than
hidden: the weights are loaded once (the registry caches the specialist instance), so the second cost
is a forward pass, not a 60 MB load. The alternative (passing the change map between steps) needs the
controller to hand artifacts between steps, which it does not do today. `execute` therefore accepts an
optional `change_map` path in `request.params` for a future planner to populate — **unused today and
documented as such**, so the hook exists without pretending the wiring does.

#### Hyperparameters and selection

| Key | Value | Source |
|---|---|---|
| `apply_type_mask` | true (serving default) | config + `vqa_specialist.py` |
| `image_size` | 256 (feature extractor default) | `vqa_specialist.py` (`change_vqa.image_size`) |
| `seed` | 42 | `PROMOTION.json` → `identity.seed` |
| `epoch_selected` | **8** | `PROMOTION.json` → `identity.epoch_selected` |
| `selected_on` | Val answer accuracy | `PROMOTION.json` → `identity.selected_on` |
| `val_answer_accuracy` | **0.700018** | `PROMOTION.json` → `identity.val_answer_accuracy` |
| `stop_reason` | `early_stopping` | `PROMOTION.json` → `identity.stop_reason` |
| `LOW_CONFIDENCE_THRESHOLD` | 0.40 (a reporting threshold, not a calibration) | `vqa_specialist.py` |

The **type mask** restricts the answer to those legal for the question type. It is on by default in
serving because the type is known from the question and an illegal answer is never right; the
evaluation reports masked and unmasked separately so the mask's contribution is visible.
`PROMOTION.json` → `verification.mask_gain` is **0.0**.

#### Training data

CDVQA annotations + SECOND (SCD) imagery (`docs/DATASETS.md` §3). The CDVQA repository publishes
**annotations only**; imagery is acquired from SECOND, name-verified **2,968 / 2,968 MATCH**, extracted
to `data/cdvqa/{im1,im2,label1,label2}/`. Val images 400; val questions **16,441**; test questions
**39,686**.

**Temporal and label semantics** — established from evidence, with honest uncertainty
(`vqa_specialist.py::TEMPORAL_ORDER_NOTE`): `T1 = pre, T2 = post`; `label1=pre, label2=post` is
**proven** (agreement 1.0000 over 2,968 scenes); `im1=pre, im2=post` is **supported statistically, not
proven**. The distinction travels in the evidence payload so a reader can see which leg the result
rests on.

**Trained externally.** The head was trained outside the repository, on an external GPU (Kaggle),
following `docs/R02_KAGGLE_TRAINING_GUIDE.md`, whose entry status was
`IMPLEMENTATION_READY_FOR_EXTERNAL_TRAINING` and whose explicit contract is: *training produces an
artifact, not a verified capability, and the run record says `TRAINED_UNVERIFIED`.* The returned
checkpoint was then **promoted** through a byte-identity gate
(`PROMOTION.json` → `state`): `before_promotion: TRAINED_UNVERIFIED`,
`after_promotion: PROMOTED`. The note is explicit: *promotion records provenance and wires the serving
path; it does not itself confer VERIFIED status — that is the maintainer's ruling.*

#### Evaluation protocol

`artifacts/change_vqa/run/PROMOTION.json` → `verification`. Verification report
`ARTIFACT_VERIFICATION.md` (2026-09-22): `checks_passed: 93`, `checks_failed: 0`,
`checks_unverified: 0`.

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
| metric ruling | `OPEN — the plan leaves the accuracy/macro-F1 interpretation owner-gated. No official aggregate metric is asserted here.` | `verification.metric_ruling` |

**The wide gap between accuracy and macro-F1 means the head is carried by common classes and performs
poorly on rare ones.** The global-majority baselines make the gap concrete: 0.311546 on `test` means a
constant predictor would score 0.31, and the head scores 0.70; but the macro-F1 of 0.378 shows the
per-class picture is far weaker than the aggregate.

#### Acceptance status

**OPEN.** No promotion/acceptance decision has been recorded. `PROMOTION.json` → `state.note` states
this explicitly. The `metric_ruling` is `OPEN`.

#### Serving behaviour — degraded mode produces no answer, on purpose

`ChangeSpecialist` degrades to an untrained detector and still emits a change map. This specialist must
**not** copy that, because the outputs are not comparable: a random change map is visibly noise,
whereas an untrained 19-way classifier still emits a fluent, confident-looking `yes`. A user cannot
tell the second from a real answer, so the untrained case returns **no answer at all** — `answer=""`,
`degraded=True`, and a warning naming exactly which piece is missing. The controller then falls through
to another attributed specialist.

There is also a **feature-spec mismatch refusal**: `feature_spec_mismatch()` compares the spec hash the
head was trained on against the spec hash this deployment produces, and refuses to answer if they
differ. The reason is measured, not hypothetical: `configs/base.yaml`'s `change:` section carries no
`checkpoint_path`, so the registry builds the change feature extractor with `checkpoint_path=None` and
gets an **untrained** STANet — while `scripts/prepare_change_vqa.py` defaults to the trained LEVIR
checkpoint. Training and serving would consume different representations, and the head would still emit
a fluent `yes`. So it refuses, and names both specs.

Confidence is **uncalibrated and says so**: `ConfidenceBreakdown.method` is `"uncalibrated"`, `raw` is
the softmax probability of the chosen answer, `components` carries the top1-top2 margin, the number of
distinct answers seen, and the normalised entropy. Nothing is fitted here — the calibration contract
(§5) is a separate artefact.

#### Limitations

- **Weak on rare classes** (macro-F1 0.378 / 0.372 vs accuracy 0.698 / 0.651).
- **Two test sets, both reported** — and the second is materially lower on accuracy (0.651 vs 0.698).
- **Ruling OPEN** — no official aggregate metric is asserted.
- **Serving needs two artefacts to be honest**: a trained head *and* a trained feature extractor; the
  shipped config wires neither by default.
- **`im1=pre, im2=post` is supported statistically but not proven** — a residual provenance risk.

---

### 3.3 `optical_sar` — CROMA-base fusion head

**Status:** `IMPLEMENTED`, `MEASURED`, ruling **OPEN**. This is the spec's #1 evaluation priority, and
it is the only workflow whose inputs are two **different sensors**.

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `optical_sar_fusion_head` | `models/manifest.json` |
| Path | `artifacts/optical_sar/fusion_head_production_v001/head.pt` | manifest |
| Bytes | 14,427,457 | manifest, `pre_registered_115_metric.json` → `head_bytes` |
| sha256 | `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab` | manifest, `checksums.sha256`, metric JSON `head_sha256` |
| HF path | `optical_sar/head.pt` | manifest |
| Architecture | CROMA-base fusion head (input_dim 2318 → hidden 512 → 19 classes) | manifest |
| Backbone | `antofuller/CROMA` (`CROMA_base.pt`, rev `0dd28e3d633b`) | manifest |
| Config hash | `78f1e3700da15aa1` | manifest |
| Parameters | `UNKNOWN — not established from the available evidence` (manifest records `null`) | manifest |

#### Architecture (`specialists/optical_sar/fusion_head.py`)

CROMA produces three 768-dimensional vectors per sample (an optical GAP, a SAR GAP, and a joint GAP).
The fusion head turns those, plus the availability masks, into a task prediction.

```
    optical_GAP      (B, 768)
    SAR_GAP          (B, 768)
    joint_GAP        (B, 768)
    optical_mask     (B,  12)     <- availability, from the sensor adapter
    sar_mask         (B,   2)     <- availability, from the sensor adapter
                     ---------
    concat           (B, 2318)
```

The frozen head architecture (`build_fusion_head`, freeze §2.5 / plan §17):

```
LayerNorm -> Linear(input_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, task_dim)
```

No activation on the output: this is a logit-producing task head, and a softmax here would be applied
twice once a loss function adds its own.

**The mask goes here, not into CROMA (finding C-1).** See §3.0.1. `CROMA_GAP_KEYS = ("optical_GAP",
"SAR_GAP", "joint_GAP")` is declared once so the rest of the package refers to one spelling.

#### Hyperparameters — every value

From `configs/base.yaml` (`croma:` and `fusion:`):

| Key | Value |
|---|---|
| `croma.checkpoint_repo` | `antofuller/CROMA` |
| `croma.checkpoint_file` | `CROMA_base.pt` |
| `croma.checkpoint_revision` | `0dd28e3d633b` |
| `croma.variant` | `base` |
| `croma.image_resolution` | 120 |
| `croma.encoder_dim` | 768 |
| `croma.optical_channels` | 12 |
| `croma.sar_channels` | 2 |
| `croma.modalities` | `[optical, sar, joint]` |
| `croma.modalities_used` | `[optical, sar, joint]` |
| `fusion.input_dim` | 2318 |
| `fusion.hidden_dim` | 512 |
| `fusion.dropout` | 0.2 |
| `fusion.num_classes` | 19 |

**Seed sweep.** Training was run as two arms (**armA**, **armB**) × five seeds (**100–104**), with a
per-arm variance report (`armA_seed_variance_report.json`). The **production head** is a distinct,
frozen artifact (`fusion_head_production_v001/head.pt`) with its own `production_head_record.json` and
a `phase12_rerun_verification.json`.

**A/B arm decision.** Made separately on `best_val_accuracy`: **A 0.837100 vs B 0.839100**, floor
**0.0285** → **Arm A retained** (owner ruling R-14,
`docs/PHASE12_115_METRIC_COMPUTED.md` §5). The metric JSON records `cache_arm: "A"`.

**Channel dropout is mandatory** (freeze §2.5). Plan §20's robustness schedule: optical is trained at
100/80/60/40 % channel availability and SAR at 100/50 %. The mask is updated to match the dropped
features — that is the entire point: dropping features while leaving the mask saying "present" would
teach the head that the mask lies. Dropped channels are zeroed, **not renormalised** — renormalising
would fabricate a scale the real missing-channel case does not have.

#### Training data

BigEarthNet (CLC-19), **single-label subset** (see the caveat below). Measured local subset: 28,000
S2 patches, 98 tiles, 12 bands per patch; full official corpus 480,038 patches. Training consumes
cached CROMA features (`fusion_features/`, `fusion_features_armB/`, ~231 MB each), which are
**reproducible and not released as model weights**.

**Measured context for the single-label policy:** single-label patches are **17.57 %** of the corpus
(96,537 of 549,488), and under this policy the rarest class survives as **1 patch** — a **59,204 : 1**
imbalance (`docs/PHASE12_LABEL_POLICY_DECISION.md` §3.0).

#### Evaluation protocol

`artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`. This is a
**separate, later, read-only** step (`scripts/eval_fusion_115.py`), deliberately **not** part of the
trainer: every `run_record.json` carries `pre_registered_metric_computed = false` and
`result_status = "PLUMBING_ONLY — fixture/loop evidence, NOT a result; the pre-registered 11.5 metric
is not computed"`. That flag is **correct about the trainer** and must not be "fixed" — the trainer
fits on `train`/`val` and never opens the test split, deliberately, so the held-out split cannot be
contaminated by the search over 10 runs, 2 arms and 5 seeds.

The tool **refuses (exit 2)** rather than guessing when: the head or cache is missing; the split is
empty; the cache `config_hash` is not `78f1e3700da15aa1` (a different experiment); or the head cannot
be built against the frozen `(B, 2318)` input contract.

Definition (per `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §4): *fusion-head accuracy and macro-F1
over the 19-class label space on the held-out split.*

#### Measured numbers

All from `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`:

| Metric | Value | Exact key path |
|---|---|---|
| **accuracy** | **0.931** | `accuracy` |
| **macro F1** | **0.434161** | `macro_f1` |
| loss | 0.254592 | `loss` |
| split | `test` | `split` |
| n scored | 4,000 | `n_scored` |
| num classes | 19 | `num_classes` |
| macro-F1 denominator | all 19 classes (absent classes contribute 0.0) | `macro_f1_denominator` |
| classes present | `[0,2,3,4,5,6,7,8,9,10,12,13,17,18]` | `classes_present` |
| classes absent | `[1,11,14,15,16]` | `classes_absent` |
| deciding statistic | **false** | `is_deciding_statistic` |
| cache arm | A | `cache_arm` |
| cache path | `artifacts/optical_sar/fusion_features/test.npz` | `cache_path` |
| cache config hash | `78f1e3700da15aa1` | `cache_config_hash` |

**Optical-SAR accuracy 0.931 must ALWAYS travel with macro-F1 0.434161.** The two are recorded side by
side in the artifact precisely so neither can be quoted alone.

Per-class F1 (`_per_class_f1`, 19 terms, indexed 0–18):

```
[1.000, 0.000, 0.428, 0.571, 0.887, 0.000, 0.000, 0.800, 0.929, 0.940,
 0.182, 0.000, 0.200, 0.438, 0.000, 0.000, 0.000, 0.877, 0.998]
```

**Two different averages come from the same per-class vector, and this is a trap:**

| Averaged over | Value |
|---|---|
| **all 19 slots** — the pre-registered definition | **0.434161** |
| the 14 present classes only | **0.589218** |

A reader who computes the second and compares it to the recorded scalar will wrongly conclude the
recorded figure is wrong. It is the 19-slot mean, and the trainer's `_macro_f1` divides by
`num_classes` by construction. Both are reproducible:
`sum(per_class)/19 = 0.43416082670034284` and
`sum(per_class[c] for c in present)/14 = 0.5892182648076082` (`docs/PHASE12_115_METRIC_COMPUTED.md`
§3.5).

**Five absent classes contribute `0.0` by definition — but two present classes also score `0.0`.**
Classes **1, 11, 14, 15, 16** are absent (their `0.0` describes no prediction); classes **5** and
**6** are **present** and score `0.0`, i.e. a **real total miss**. So the honest reading is *both*: part
of the low macro-F1 is populational, **and** there is genuine per-class failure. High accuracy with a
wide per-class spread is the signature of prediction dominated by frequent classes.

Measured test-set label distribution (`docs/PHASE12_115_METRIC_COMPUTED.md` §3.4):

```
[13, 0, 63, 39, 846, 4, 1, 6, 453, 246, 4, 0, 9, 18, 0, 0, 0, 34, 2264]
```

14 of 19 classes present; five have zero samples; top-to-bottom ratio **2,264 : 1** (class 18 vs class
6). The majority class holds 2,264 / 4,000 = **0.566**, and the head scores **0.931** — so the accuracy
is not a constant predictor. Per-class recall on the held-out split (selected rows): class 18 n=2264
recall **0.997**; class 9 n=246 recall **0.951**; class 17 n=34 recall **0.941**; class 8 n=453
**0.905**; class 4 n=846 **0.809**; class 3 n=39 **0.410**; class 12 n=9 **0.111**; classes 5 and 6
recall **0.000**.

**Four independent verification checks** (`docs/PHASE12_115_METRIC_COMPUTED.md` §3): (1) calling
`evaluate_fusion_head` directly returns `n=4000, accuracy=0.931, macro_f1=0.43416082670034284`;
(2) rebuilding the input tensor by hand — `concat([optical_gap, sar_gap, joint_gap])` then
`concat([…, optical_mask, sar_mask])`, widths `(4000,768)×3 + (4000,12) + (4000,2) = 2318` — gives
0.931 again (this checks feature *ordering*); (3) `sklearn.metrics` `accuracy_score = 0.931`,
`f1_score(average="macro", labels=range(19), zero_division=0) = 0.43416082670034284`; (4) the
per-class breakdown above.

#### Acceptance status

**OPEN.** `is_deciding_statistic: false`. The artifact's own `advisory` field states: *"This tool
reports ONE head's held-out accuracy and macro-F1. It selects no head, ranks nothing and compares no
arms. Whether this constitutes a Phase 12 pass is the owner's ruling."* Ruling 5 attaches **no
numerical threshold**; ruling 4 requires it be described as a **single-label subset evaluation, never
as a multi-label BigEarthNet/reBEN result**.

#### ⚠️ The caveat that governs how these numbers may be used

The cache metadata records `label_policy = require_single_label`, `n_skipped_by_policy = 0`. reBEN
v2.0 is a **multi-label** corpus while the frozen head is a **single-label 19-class softmax** trained
with `cross_entropy`. The extraction therefore used policy (b) — restrict to single-label patches —
which preserves the frozen architecture and the 19-class space exactly but **changes the evaluation
population**.

So this metric **may not** be presented as: a multi-label BigEarthNet/reBEN result; comparable to
published BigEarthNet numbers (almost all multi-label); or a statement about all 19 classes (**5 have
no test samples here**). It **may** be presented as: *the pre-registered 11.5 metric, as computed under
the single-label extraction policy the frozen architecture requires, on the held-out split.*

#### Limitations

- **Accuracy is carried by common classes** (0.931 vs macro-F1 0.434161; two present classes at 0.0).
- **The live service returns a bare class index** (`class_18`), not a human-readable label
  (`_label_for` falls back to `f"class_{index}"` when `class_labels` is empty).
- **Single-label subset** — not comparable to multi-label BigEarthNet numbers.
- **Ruling OPEN**; no threshold attached.
- **The confidence path floors to 0.0 when there is no prediction or no trained head** — a signal
  *gap*, not a weak signal. Measured with no checkpoint: `raw = 0.0`,
  `components = {optical_confidence: 0.333, sar_confidence: 1.0, cross_modal_agreement: 0.0,
  has_croma: 0.0, trained_head: 0.0, no_prediction: 1.0}`, `degradation_reason: "CROMA is not loaded"`.
- **Two assets must be the right two.** `validate_request` rejects 0, 1, 3, and also **two of the same
  kind** — a plausible mistake that, if accepted, would place a second optical image into the 2-channel
  SAR slot, zero-fill, run CROMA, and answer about one modality while claiming to fuse two.

---

### 3.4 `grounding` — RemoteCLIP grounding head

**Status:** `IMPLEMENTED`, `MEASURED` — **two protocols** and **two decode variants**.

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `grounding_head` | `models/manifest.json` |
| Path | `artifacts/grounding/remoteclip_grounding_v001/head.pt` | manifest |
| Bytes | 12,639,041 | manifest |
| sha256 | `93432f7034be91a8ffd9c1a84e3eeec00bed7832c043fe7f83d2be230284c6bb` | manifest, `checksums.sha256` |
| HF path | `grounding/head.pt` | manifest |
| Architecture | RemoteCLIP ViT-B/32 grounding head (feature_dim 2048, hidden 512) | manifest |
| Backbone | `chendelong/RemoteCLIP` (`RemoteCLIP-ViT-B-32.pt`, rev `bf1d8a3ccf2d`) | manifest |
| Config hash | `78f1e3700da15aa1` | manifest |
| **Parameters** | **1,052,677** | `docs/PHASE8_GROUNDING_HEAD_DECISION.md` |

#### Architecture (`specialists/grounding/head.py`)

A trainable head over the **frozen** RemoteCLIP ViT-B/32 encoder. Per-cell feature is
`concat([patch, text, patch·text, global_pool]) = 4 × 512 = 2048` (enforced at config load — §3.0.2).

```
FROZEN INPUTS (measured, docs/PHASE7_GROUNDING_CONTRACT.md):
    patch tokens : (B, 49, 512) at 224px   -- 7x7 grid, projected dim 512
    text emb     : (B, 512)
```

The head is `proj: Linear(2048, 512)`, `norm: LayerNorm(512)`, `drop: Dropout(0.10)`,
`out: Linear(512, 5)` — five outputs `[tx, ty, tw, th, objectness]`. Forward: `proj → GELU → LayerNorm
→ Dropout → out`.

**Small-std init, and objectness bias low** (`_init_weights`): `proj.weight ~ N(0, 0.02)`,
`proj.bias = 0`, `out.weight ~ N(0, 0.01)`, `out.bias = 0` except `out.bias[4] = -2.0`. With one
positive cell in 49 the task starts 1:48 imbalanced; a bias of −2.0 starts the objectness prior near
0.12, closer to the truth than 0.5, and stops the first epochs being spent un-learning a saturated
sigmoid.

**Box parameterisation — cell-relative (YOLO-style).** Cell `(r, c)` of a `grid_h × grid_w` map
predicts `[tx, ty, tw, th, obj]`:

```
cx = (c + sigmoid(tx)) / grid_w
cy = (r + sigmoid(ty)) / grid_h
w  = sigmoid(tw)
h  = sigmoid(th)
box = clip((cx - w/2, cy - h/2, cx + w/2, cy + h/2), 0, 1)
```

Cell-relative rather than absolute because 7×7 is coarse: an absolute regressor must learn 49 separate
mappings onto the same global coordinates; a cell-relative one only learns a local offset. After
clipping, `enforce_order` guarantees `x2 ≥ x1` and `y2 ≥ y1` — without it a decoded box can be
inverted, which makes IoU zero and silently kills the gradient.

**Positive assignment.** Exactly one cell per target: the one containing the ground-truth box
**centre** (standard single-stage-detector convention, YOLO/FCOS). Ties go to the earlier cell so the
assignment is deterministic. This makes the 1-of-49 confidence imbalance explicit and therefore
correctable.

**NMS** is pure torch (no torchvision dependency), so the deployed Space needs no extra package and
the behaviour is identical on CPU and GPU.

**Loss** (`grounding_loss`): `box = L1(pos_boxes, targets)` (weight 0.5),
`giou = 1 - GIoU(pos_boxes, targets)` (weight 0.3), and `conf = BCE_with_logits(obj_logits,
conf_targets, weight=weights)` (weight 0.2) where the single positive cell is up-weighted by
`positive_confidence_weight = 20.0`. Box and GIoU are computed on the **decoded** coordinates of the
single positive cell per sample, so the loss is computed on exactly the coordinates the metric
measures. GIoU is used alongside L1 because L1 alone is scale-blind. `LossBreakdown` keeps the
components separate so a collapse in one is visible in the logs.

#### Hyperparameters — every value

From `configs/base.yaml` (`grounding:`, `grounding_head:`, `grounding_training:`):

| Key | Value |
|---|---|
| `grounding.checkpoint_repo` | `chendelong/RemoteCLIP` |
| `grounding.checkpoint_file` | `RemoteCLIP-ViT-B-32.pt` |
| `grounding.checkpoint_revision` | `bf1d8a3ccf2d` |
| `grounding.model_name` | `ViT-B-32` |
| `grounding.image_size` | **224** |
| `grounding.resolution_frozen` | true |
| `grounding.nms_iou` | 0.50 |
| `grounding.max_candidates` | 20 |
| `grounding.confidence_threshold` | 0.40 |
| `grounding.benchmark_box_scale` | 100.0 |
| `grounding.coordinate_system` | `normalized_0_1` |
| `grounding.encoder_projected_dim` | 512 |
| `grounding_head.hidden_dim` | 512 |
| `grounding_head.dropout` | 0.10 |
| `grounding_head.feature_dim` | 2048 |
| `grounding_head.positive_confidence_weight` | **20.0** |
| `grounding_head.decode` | `cell_relative` |
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

**Why `positive_confidence_weight = 20.0`.** Objectness BCE sees ~1 positive cell out of 49.
Unweighted, the optimum is "no object" everywhere; the weight is what stops that collapse.

#### Training data

VRSBench. The head trains on the train split and is evaluated on the VRSBench **eval** split, n =
**16,159** records, scored once, no sampling. VRSBench stores boxes normalised to **0–100**; this
project stores boxes normalised to **0–1**, via the declared `benchmark_box_scale: 100.0` so the
conversion cannot be applied twice or forgotten.

**Validation trajectory** (20 epochs, CPU, ~37 s/epoch) — `docs/PHASE8_GROUNDING_HEAD_DECISION.md`:

```
epoch   1/20  loss 0.4284  box 0.0641  giou 0.7948  conf 0.7897  val_iou 0.0436
epoch   5/20  loss 0.3864  box 0.0489  giou 0.7581  conf 0.6725  val_iou 0.0673
epoch  10/20  loss 0.3653  box 0.0446  giou 0.7313  conf 0.6179  val_iou 0.0831
epoch  15/20  loss 0.3488  box 0.0414  giou 0.7062  conf 0.5811  val_iou 0.0926
epoch  20/20  loss 0.3419  box 0.0401  giou 0.6937  conf 0.5685  val_iou 0.0943
```

All four loss components move monotonically in the right direction. `conf` falls from an
over-confident 0.79 as the positive-cell weighting corrects the objectness prior; `box` and `giou` fall
together, which is the healthy pattern (a falling `box` with a rising `giou` would mean the box is
drifting in size). **Validation IoU (0.0943 at epoch 20) is much lower than eval IoU (0.2566).** These
are not comparable and the gap is expected: validation is a held-out slice of the **train** split,
which has noisier ground truth (29.5 % of its boxes are out-of-range and filtered, and the surviving
ones come from a different annotation pass). This is a property of VRSBench, not a bug.

#### Evaluation protocol — two protocols × two decode variants

`artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` (top_k = 20, the config
default) and `…/eval_result_matched6.json` (top_k = 6, decode-matched to the baseline). Both record
`n_eval_records: 16159`, `resolution: 224`, `grid: 7`, `limited_run: false`,
`frozen_config_evaluation: true`, device `cpu`, torch `2.14.0+cpu`, and
`config_hash: 78f1e3700da15aa1`, `config_drift: false`.

**The `head_decode` block is recorded in each artifact** — `config_default_top_k: 20`, `nms_iou: 0.5`,
`score_threshold: 0.4`, `top_k: 20` (canonical) or `top_k: 6` (matched6). Without it the artifact could
not say which setting produced its number, and a re-run at the config default would silently yield a
different figure.

#### Measured numbers

**The trained head, under the two protocols:**

| Protocol | mean best IoU | recall@0.5 | Artifact key path |
|---|---|---|---|
| canonical (head threshold decode, top_k = 20) | **0.2838** | **0.2198** | `results.head_threshold.mean_best_iou` / `results.head_threshold.recall.0.50` |
| matched6 (head threshold decode, top_k = 6) | **0.2566** | **0.1938** | `results.head_threshold.mean_best_iou` / `results.head_threshold.recall.0.50` |

**Two decode variants, reported for completeness** (a reviewer must see the whole grid, not one cell):

| Variant | mean best IoU | recall@0.10 | recall@0.25 | recall@0.50 |
|---|---|---|---|---|
| head argmax decode (canonical) | **0.1215** | 0.3183 | 0.2088 | 0.0795 |
| zero-shot matched (no trained head, canonical) | **0.0972** | 0.3298 | 0.1188 | 0.0234 |

**Full canonical grid** (`eval_result_canonical.json` → `results`):

| Strategy | `mean_best_iou` | `recall.0.10` | `recall.0.25` | `recall.0.50` | `latency_ms_per_image` | `seconds` |
|---|---|---|---|---|---|---|
| `head_argmax` (strategy `argmax`) | 0.1215 | 0.3183 | 0.2088 | 0.0795 | 0.655 | 10.6 |
| `head_threshold` (strategy `threshold`, top_k 20) | 0.2838 | 0.6882 | 0.5047 | 0.2198 | 2.205 | 35.6 |
| `zero_shot_matched` | 0.0972 | 0.3298 | 0.1188 | 0.0234 | — | 17.9 |

**Full matched6 grid** (`eval_result_matched6.json` → `results`): `head_argmax` identical (0.1215);
`head_threshold` (top_k 6) 0.2566 / 0.6315 / 0.4545 / 0.1938 at 2.158 ms; `zero_shot_matched`
identical (0.0972). Each artifact also carries `phase7_reference: {mean_best_iou: 0.0972,
recall_at_0.50: 0.0234, source: "docs/PHASE7_RESOLUTION_DECISION.md"}`.

**Decode-matched delta (head threshold top_k = 6 vs zero-shot): +0.1594 IoU, +0.1704 Recall@0.50.**
The pre-registered bar was `MIN_IMPROVEMENT_IOU = 0.02`; the measured margin is **8× the bar**.

**`head_argmax` vs zero-shot is NOT apples-to-apples.** 1 box against ~6 boxes flatters the head,
because mean best IoU takes the max over predictions. It is reported because it is the number directly
comparable to the Phase 7 resolution experiment's zero-shot argmax, not because it decides anything.

**A defect this phase had to fix first** (`docs/PHASE8_GROUNDING_HEAD_DECISION.md`). The evaluation
script originally built its own single-box baseline with `argmax_candidate`, while Phase 7 measured the
baseline through `ground_phrase` (threshold box + up to 5 local maxima). Same 16,159 records, same
metric, same cached features — **different decode**: Phase 7 via `ground_phrase` = 0.0972, eval via
`argmax_candidate` = 0.0092, a 10× gap. The eval printed `head beats zero-shot: True (+0.1123)` when
the matched comparison was +0.0243. Both cleared the bar, but only the second is a claim about the head
rather than about the decode. **Fix:** the decode was extracted into one function both paths call —
`specialists/grounding/inference.py::decode_candidates_from_features` — so there is exactly one place
that turns similarity into boxes. Verification that the fix is real, not cosmetic: re-running the eval
reproduced the Phase 7 number exactly (`|difference| = 0.0000`).

**A second defect: candidate count changes the number.** Mean best IoU is a max over predictions, so
emitting more boxes raises it mechanically. The head's threshold decode defaulted to `max_candidates:
20` while the baseline emits 5.99 — an uncontrolled asymmetry in the head's favour. Fix:
`--head-top-k` and `--head-score-threshold` overrides, plus the recorded `head_decode` block. Measured
cost of the cap: top_k 20 → 0.2838 (+0.1866 vs zero-shot); top_k 6 → 0.2566 (+0.1594). Capping to the
baseline's own budget costs 0.027 IoU — **the win survives.**

#### Acceptance status

`MEASURED`. Phase 8 verdict: **the trained head beats the zero-shot baseline**; Phase 8 is done. The
head is **not yet wired into the specialist** as of the Phase 8 decision record
(`specialists/grounding/inference.py` still exposes only the zero-shot path), and that integration step
must not change the zero-shot module's behaviour because the resolution experiment depends on it.

#### Limitations

- **Absolute IoU is low.** 0.2838 / 0.2566 is not "solved".
- **Protocol-sensitive.** 0.2838 (canonical) vs 0.1215 (argmax) — absolute values depend on decode.
- **Localisation floor.** At 224 each token covers 1/7 (~0.143) of the image width; at 448 it is 1/14.
- **`head_threshold` at top_k = 20 was not tuned.** 20 is the config default, not a
  validation-selected optimum. Selecting it on eval would be benchmark tuning.
- **Nothing about the hidden distribution.** VRSBench is overhead optical; the hidden set is
  Cartosat-2S + RISAT.
- **Not a system-level result.** "Beats the zero-shot baseline on VRSBench eval" ≠ "performs well on
  the hidden ISRO/SAC set".

---

### 3.5 `router` — intent adapter over frozen MiniLM

**Status:** `IMPLEMENTED`, `MEASURED`, **TEST NOT RUN**.

A **~50,822-parameter adapter** over the frozen MiniLM encoder. Because the encoder is frozen,
embeddings are cached and the adapter trains on cached vectors — no GPU required (measured: 20 epochs /
4,096 vectors in 0.28 s on CPU).

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `router_adapter` | `models/manifest.json` |
| Path | `artifacts/router/router_adapter_v001/adapter.pt` | manifest |
| Bytes | 211,961 | manifest |
| sha256 | `8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c` | manifest, `checksums.sha256` |
| HF path | `router/adapter.pt` | manifest |
| Architecture | task/modality adapter over frozen MiniLM embeddings (~50,822 params) | manifest |
| Backbone | `sentence-transformers/all-MiniLM-L6-v2` (rev `1110a243fdf4`) | manifest |
| Config hash | `78f1e3700da15aa1` | manifest |
| Parameters | `UNKNOWN — not established from the available evidence` (manifest records `null`) | manifest |

> **Parameter-count discrepancy — flagged, not smoothed over.** `specialists/router/adapter.py`'s
> docstring and `models/manifest.json` both state **~50,822** parameters. The Phase 4 completion
> report (`docs/PHASE4_ROUTER_REPORT.md`) states **51,725** adapter parameters on a frozen
> 22,713,216-param encoder and an artifact size of 223.8 KB, whereas the released artifact is
> 211,961 bytes. These two figures were produced at different times and have not been reconciled.
> The release documents the **manifest** figure (50,822) as the shipped number; which figure is
> authoritative is
> `UNKNOWN — not established from the available evidence`.

#### Architecture (`router/adapter.py`)

The only trainable part of the router. It sits on top of the frozen MiniLM embedding and emits **five
heads**:

```
embedding (384)
    |
LayerNorm
    |
Linear(384 -> hidden_dim)      default hidden_dim = 128
    |
GELU
    |
Dropout
    |
    +--> task_head            Linear(hidden, 6)
    +--> modality_head        Linear(hidden, 4)
    +--> temporal_head        Linear(hidden, 1)   logit
    +--> spatial_head         Linear(hidden, 1)   logit
    +--> language_head        Linear(hidden, 1)   logit
```

`_init_weights` uses small-std init (`std = 0.02`) with zero bias on every head, keeping the initial
sigmoid near 0.5 — without it the binary heads can start saturated and BCE gradients vanish before the
task head has learned anything useful. `forward` asserts the input is 2-D and that
`embeddings.shape[1] == input_dim`, and requires embeddings already detached from the frozen encoder
(the adapter does not back-propagate into MiniLM).

#### Hyperparameters — every value

From `configs/base.yaml` (`router:`):

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
| `training.hard_negatives_to_test` | true |

**Splits are by GROUP (template / hard-negative family), never by example.** Template-generated
queries are near-duplicates; splitting by example would put `"Show me the water body."` in train and
`"Show me the road."` in val — one token apart — and report a fake accuracy. Hard-negative families
are placed in the **test** split so their accuracy measures generalisation, not memorisation.

#### Training data

A **synthetic** corpus: 576 examples, 54 groups, hand-written and templated
(`threshold_sweep_val.json`: `corpus_total: 576`, `corpus_groups: 54`, `corpus_limited: true`). Split
sizes: `train 410`, `val 86`, `test 80`. The plan's minima are `plan_min_val_queries: 500` and
`plan_min_hard_negatives: 100`; the corpus is well below both, and the artifact says so.

#### Evaluation protocol and measured numbers

`artifacts/router/threshold_sweep_val.json`. The sweep iterates thresholds 0.50 → 0.99 and records
coverage, covered-task accuracy, fallback rate and `n_covered` per threshold. `select_by:
"covered_accuracy"`; `selected.threshold: 0.76` (coverage 0.790698, covered-task accuracy 1.0);
`shipped_threshold: 0.70` (coverage 0.848837, covered-task accuracy 0.972603).

| Field | Value | Exact key path |
|---|---|---|
| **overall ungated task accuracy** | **0.965116** | `overall_ungated_accuracy` |
| split | `val` | `split` |
| n val | **86** | `n_val` |
| n val examples scored | 86 | `n_val_examples_scored` |
| n test examples scored | **0** | `n_test_examples_scored` |
| `test_split_touched` | **false** | `test_split_touched` |
| `corpus_limited` | true | `corpus_limited` |
| `hard_negatives_in_val` | **0** | `hard_negatives_in_val` |
| `val_min_support` | 8 | `val_min_support` |
| `val_task_counts` | `{caption: 8, change: 20, grounding: 14, optical_sar: 10, unsupported: 19, vqa: 15}` | `val_task_counts` |
| `split_sizes` | `{train: 410, test: 80, val: 86}` | `split_sizes` |
| `delta_vs_shipped.coverage` | −0.0581 | `delta_vs_shipped.coverage` |
| `delta_vs_shipped.covered_task_accuracy` | 0.0274 | `delta_vs_shipped.covered_task_accuracy` |
| adapter encoder params | 22,713,216 | `adapter_encoder.parameters` |
| `adapter_config_hash` | `615478910dc266bf` | `adapter_config_hash` |
| `config_hash` | `78f1e3700da15aa1` | `config_hash` |

**This number is (a) validation-only, (b) ungated, and (c) small (n = 86).** The artifact's own note is
blunt: *"corpus-limited: val n=86 vs plan >=500. This is NOT a calibration — the corpus is synthetic
and too small (min per-class support 8, caption) and val carries 0 hard negatives (hn_* families are
held out to TEST by design). Selecting a threshold here yields a justified default, not a calibrated
value. The corpus was NOT padded with generated queries. Backlog P1-9's 'n=80' is the TEST split; the
sweep target is val n=86. **The test split was NOT touched.**"*

**The router TEST split was NOT RUN.** Do not read 0.965116 as a test result. The sweep scored 86
validation examples and zero test examples, and `test_split_touched` is `false`.

> **Historical note, recorded so the two are not conflated.** The Phase 4 completion report
> (`docs/PHASE4_ROUTER_REPORT.md`, verified 2026-09-16, Gate 2 PASS) records an **earlier** gate-2
> evaluation that included a test column — test task accuracy **0.975**, macro F1 0.976, per-class test
> recall (vqa 1.000/n=15, caption 0.846/n=13, grounding 1.000/n=17, change 1.000/n=13, optical_sar
> 1.000/n=12, unsupported 1.000/n=10) and hard-negative accuracy **0.800**. That evaluation predates
> the shipped threshold sweep, and its own standing caveats are that the corpus is synthetic and that
> *"the 0.975 headline is partly earned on templates the split kept in training. Treat the router as
> working, not as benchmarked."* The release's position for the shipped artifact is **TEST NOT RUN**;
> the Phase 4 numbers are retained as a historical record, not promoted to a release benchmark.

#### Acceptance status

`MEASURED`; the headline is validation, ungated, n = 86. No test result is claimed.

#### Limitations

- **Corpus-limited** (n = 86 val; plan minimum 500; hard negatives in val = 0 by design).
- **Not a calibration** — the threshold is a justified default, not a calibrated value.
- **Synthetic corpus** — 576 hand-written/templated examples.
- **`caption` has the smallest support** (8 in val), the tightest per-class floor.
- **Known misroutes** are catalogued in [`LIMITATIONS.md`](LIMITATIONS.md) §2 (e.g. *"What is the new
  runway?"* reads `change`, not `vqa`; *"How much built-up area was added?"* under-triggers to `vqa`).
- **Threshold 0.70 is uncalibrated** — it gates the lexical fallback.

---

### 3.6 `vlm` — SmolVLM LoRA adapter (USABLE_VERIFIED, ACCEPTANCE-REJECTED)

**Status:** `IMPLEMENTED`, `MEASURED`, **ACCEPTANCE-REJECTED**.

> This is §3.6 because [`TRAINING.md`](TRAINING.md) §7 cross-references it. The section number is kept
> stable so that link resolves.

A PEFT LoRA adapter on frozen `HuggingFaceTB/SmolVLM-500M-Instruct`. **The artifact's status is
`CLOSED` with headline `ACCEPTANCE-REJECTED`.** This is not a contradiction — it is the project's
central truthfulness distinction: **`USABLE_VERIFIED`** (the adapter demonstrably works; the metrics
are real and reproducible) and **`ACCEPTANCE-REJECTED`** (it is *not accepted* for production
promotion). **USABLE ≠ ACCEPTED.** The deployed caption/VQA path therefore uses the **unadapted**
SmolVLM.

#### Identity

| Field | Value | Source |
|---|---|---|
| Manifest `id` | `vlm_lora_adapter` | `models/manifest.json` |
| Path | `.scratch/phase6_real_adapter/phase6_adapter/adapter_model.safetensors` | manifest |
| Bytes | 34,798,048 | manifest |
| sha256 | `07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e` | manifest, `checksums.sha256` |
| HF path | `vlm/adapter_model.safetensors` | manifest |
| Architecture | PEFT LoRA (r=16, alpha=32, dropout=0.05) on text_model projections | manifest |
| Base model | `HuggingFaceTB/SmolVLM-500M-Instruct` (rev `a7da5b986cb5`) | manifest, closure |
| `kind` | `bigearthnet_smolvlm_lora` | closure → `production_adapter.kind` |
| **tree sha256** | `5c6b86317d1e65962702dc9e377009b3df41cc13de1b15bceccb70ad977775e7` | closure → `adapter_tree_sha256` |
| **file sha256** | `07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e` | closure → `weights_file_sha256` |
| Trainable params | **8,683,520** (1.6823 % of 516,165,824) | closure → `trainable_params` |
| Config hash | `78f1e3700da15aa1` | manifest |
| `retrained_for_closure` | **false** | closure |
| `modified_for_closure` | **false** | closure |

#### Architecture and target modules

LoRA, `peft_type=LORA`, `r=16`, `alpha=32`, `dropout=0.05`, targeting
`model.text_model.*.{q,k,v,o,gate,up,down}_proj` — `lora_target_module_count: 224`.

**The vision-tower hazard did not occur.** `trainable_subtrees` is exactly
`{"model.text_model": 8683520}`; the vision model (86,433,024) and connector (11,796,480) appear in
`frozen_params`, which are:

| Frozen subtree | Parameters |
|---|---|
| `model.connector` | 11,796,480 |
| `model.text_model` | 361,944,000 |
| `model.vision_model` | 86,433,024 |
| `other` | 47,308,800 |

The `trainable_params` figure was measured two independent ways (closure → `trainable_params.how`):
summing `numel` over every tensor in `adapter_model.safetensors` read from the safetensors header
(no model load), and loading through the production path and reading `sum(p.numel())` then subtracting
the manifest frozen-params sum. Both give 8,683,520; `manifest_matches_measurement: true`.

#### Hyperparameters — every value

From `configs/base.yaml` (`training:`) and the closure record:

| Key | Value |
|---|---|
| `training.precision` | **fp16** (finding C-6: T4 is SM 7.5 → fp16, NOT bf16) |
| `training.vlm_batch_size` | 2 |
| `training.vlm_gradient_accumulation` | 8 |
| `training.vlm_learning_rate` | 0.0002 |
| `training.vlm_epochs` | 1 |
| `training.lora_rank` | 16 |
| `training.lora_alpha` | 32 |
| `training.lora_dropout` | 0.05 |
| `training.weight_decay` | 0.01 |
| `training.warmup_ratio` | 0.05 |
| `training.gradient_checkpointing` | true |
| `training.save_every_steps` | 500 |
| PEFT version | **0.19.1** |
| Seed | 42 |
| `precision_recorded` | fp16 |

#### Training data

BigEarthNet instruction pairs (the same 28,000-patch local subset used for the fusion head), on an
external GPU (Kaggle T4-class). This is the only artifact that requires a GPU to train.

#### Evaluation protocol

`artifacts/vlm/phase6_closure.json` (generated by `scripts/phase6_close.py`, which reads the evidence
rather than restating it). The evaluation is on a **frozen 1,000-question subset** of the test split.
The subset's identity was proven **without a model** (Gate A″): `available_per_split {val: 6750, test:
7772}`, subset `n = 1000`, and the test per-class counts an **exact** match to the recovery subset
(19 classes, sum 1000), `config_hash 78f1e3700da15aa1` identical to Run 1's.

Gate D reproduced Run 1's adapted-test control **exactly** — loading the local adapter and evaluating
it returned `exact_match 0.963`, `n 1000`, confusion `{tp:500, fp:19, tn:463, fn:18}`,
`f1 0.9643201542912246` — identical to Run 1's recorded values, which proves the local artifact *is*
Run 1's adapter and that CPU/fp32 reproduces the Kaggle T4 endpoint.

#### Measured numbers — the "usable" side

`artifacts/vlm/phase6_closure.json` → `why_usable_verified.adapted_test`:

| Metric | Value | Exact key path |
|---|---|---|
| **exact_match** | **0.963** | `why_usable_verified.adapted_test.exact_match` |
| **F1** | **0.96432** | `why_usable_verified.adapted_test.f1` |
| n | 1000 | `…adapted_test.n` |
| precision | 0.963391 | `…adapted_test.precision` |
| recall | 0.965251 | `…adapted_test.recall` |
| tp / fp / tn / fn | 500 / 19 / 463 / 18 | `…adapted_test.confusion` |
| aggregate test delta | **+49.50 pp** (46.80 → 96.30 pp) | `why_usable_verified.aggregate_test_delta_pp` |

Per-class accuracy on the adapted test subset ranges from 0.8 (Agro-forestry areas, n=20) to 1.0
(several classes); `Mixed forest` 0.878788 (n=33), `Permanent crops` 0.862069 (n=29), `Broad-leaved
forest` 0.898551 (n=69), `Pastures` 0.916667 (n=132), `Marine waters` 1.0 (n=329).

#### Measured numbers — the "rejected" side

`artifacts/vlm/phase6_closure.json` → `why_acceptance_rejected`. Decision split: **test**. Rule
version: **v002**.

| | |
|---|---|
| **V1** | **passes** — test `46.80 → 96.30 pp`, delta **+49.50 pp** (required ≥ +5.00) |
| **V2** | **fails** — **Mixed forest**: `n=33`, `100.00 → 87.8788 pp`, `drop 12.1212 pp`, `lost_questions 4`, `z 2.1335` |

It fails **both** halves of v002 (`lost ≥ 4` **and** `z ≥ 1.96`). Per item V, a complete run that fails
V2 is `REJECTED`.

Thresholds used (`why_acceptance_rejected.thresholds_used`): `accept_min_delta_pp: 5.0`,
`accept_min_delta_pp_ceiling: 2.0`, `ceiling_baseline_pp: 95.0`, `max_class_drop_pp: 1.0` (legacy
v001), `min_class_questions: 20`, `test_val_disagreement_pp: 10.0`; `v2_criterion`: `class_drop_z:
1.96`, `min_class_drop_questions: 4`, `min_class_questions: 20`, with
`se_formula: "sqrt((p_b*(1-p_b) + p_a*(1-p_a)) / n)"` and `z_formula: "(baseline - adapted) / se"`.

**The rejection is narrow.** `n_classes_failed: 1`, `n_classes_held: 5`, `n_classes_improved: 11`,
`n_classes_total: 19`. The next-worst class (Inland wetlands, −6.6667 pp, n=30) lost only 2 questions
and therefore sits below V2's materiality floor.

**It is not a split artefact.** The same class also degraded on **val** in Run 1 (drop 6.4516 pp,
n=31) — the very value that motivated v001's flag. The adapter hurts Mixed forest on **both** splits,
so this is a property of the adapter, not an accident of one subset. Mixed forest also sits at a
100.00 pp baseline on test, so any loss is a drop from the ceiling.

**Residual risk, reported not resolved** (`residual_risk`): the verdict rests on **4 questions in one
class of 33** — the unfloored minimum-size exposure recorded at `PHASE6_AUDIT_AND_CONTRACT.md` §8.6.
With no `n ≥ N` floor in V2, a 33-question class can flip the verdict of a run whose aggregate endpoint
improved by 49.5 pp.

**BERTScore is unavailable**, not zero: `thresholds_used.bertscore.available: false`, reason
`roberta-large` is not in the local HuggingFace cache. BLEU/ROUGE are **excluded** because target
answers are one token and they are meaningless at that length.

#### Both rejection records are preserved

Neither is rewritten. A future reader must be able to see what each rule said, on which split, at the
time it said it (`preserved_records`):

| Record | Rule | Split | Verdict | Source |
|---|---|---|---|---|
| Run 1's own manifest | `v001` | val | **`REJECTED`** (3 classes) | Run 1's original manifest, preserved verbatim as `preserved_records.v001_val_rejected` in `phase6_closure.json` |
| Test-split adjudication | `v002` | **test** (independent) | **`REJECTED`** (1 class) | `artifacts/vlm/run1_test_recovery/test_adjudication.json` |
| Recovery manifest | `v002` | val | `ACCEPTED` — **not final acceptance** | `artifacts/vlm/run1_test_recovery/run_manifest.json` |

**v001 on val** failed three classes — Mixed forest (6.4516 pp, n=31), Transitional woodland shrub
(9.375 pp, n=32), Agro-forestry areas (4.6512 pp, n=43) — with `val_delta_pp 42.0` and
`test_delta_pp null`. **The val-split `ACCEPTED` under v002 is recorded for completeness only and is
NOT final acceptance** — it decides on the *same val subset that motivated v002*, which
`docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §7.4 condition 3 forbids. On data that did not motivate the
relaxation, the run does not pass.

#### Acceptance status — and what closure does NOT claim

`status: CLOSED`; `headline: "Phase 6 is closed. The Run 1 LoRA adapter is promoted to the production
VLM adapter: USABLE and VERIFIED, but ACCEPTANCE-REJECTED."` The record keeps verified and accepted
separate: *"'Verified' answers: is this artifact the one we trained, and does it work? 'Accepted'
answers: did it clear the bar predeclared before we looked? Both are true, and they are different
questions."*

`what_closure_does_not_claim`:

- It does **not** claim Run 1 was accepted — it was rejected by v001 on val and by v002 on the
  independent test split.
- It does **not** claim the Mixed forest regression is resolved.
- It does **not** claim a new adapter exists or is planned.
- It does **not** alter the v002 rule or its verdict.

**Forward rule** (`forward_rule`): any future improved adapter MUST be a **new** experiment/version. It
MUST NOT rewrite, amend, or supersede Run 1's records; it must not retrain or modify the Run 1 adapter
in place; it must not report a new adapter's metrics under Run 1's identity; it must not edit v001's or
v002's recorded verdicts to match a later outcome. Precedent: v002 itself followed this rule.

#### How it is enabled, and the two traps

The adapter is attached via an **environment setting, not a code change**:
`specialists/vqa/model.py` resolves it in the order explicit `adapter_path` argument →
**`SATQUERY_VLM_ADAPTER`** (`ADAPTER_ENV_VAR`) → none, then attaches it with
`PeftModel.from_pretrained`. A load failure raises rather than silently serving the base model: an
inference result attributed to an adapter that did not actually attach is worse than a hard failure.

Two traps a future reader will hit (`known_traps`):

1. **`adapter_sha256` names two different values.** `training/vlm/artifact.py` computes a **tree hash**
   over the `{relpath: sha256}` weight map (`5c6b8631…`), while
   `specialists/vqa/model.py::_adapter_sha256` computes the **file** sha256 of
   `adapter_model.safetensors` (`07c76a75…`). Recomputing one and comparing it to the other yields a
   false "the artifact was altered" conclusion.
2. **The promoted adapter is NOT `checkpoint-2000`.** The three weight files have three different
   digests: top-level `07c76a75…`, `checkpoint-1500` `7273588e…`, `checkpoint-2000` `bf249943…`. So
   "just use the last checkpoint" is not equivalent to this artifact.

#### Limitations

- **Acceptance-rejected** — the deployed path uses the unadapted model.
- **Residual risk** — the verdict rests on 4 questions in one class of 33.
- **`Mixed forest` regression unresolved**; it degrades on both val and test.
- **The path lives under `.scratch`** — `.gitignore` excludes `artifacts/`, `checkpoints/` and
  `*.safetensors`, so the ~105 MB local directory is not committed; a future cleanup could remove it.
  It is reconstructible from `phase6_realbundle.zip` and verifiable against the two digests above.
- **BERTScore not computed** (unavailable offline); BLEU/ROUGE excluded as meaningless at one-token
  answers.

---

## 4. Rejected, deferred and open model decisions

| Decision | Outcome | Evidence |
|---|---|---|
| Grounding image resolution 448 vs 224 | **224 chosen; 448 REJECTED** | `docs/PHASE7_RESOLUTION_DECISION.md`; §4.1 below |
| VLM adapter promotion | **REJECTED** | metrics usable, acceptance rejected (§3.6) |
| Calibration | **kept but ineffective** | ECE worsened (§5) |
| Optical-SAR ruling | **OPEN** | no decision recorded; `is_deciding_statistic: false` |
| Change-VQA ruling | **OPEN** | `verification.metric_ruling` |
| CROMA constructor (`use_croma.PretrainedCROMA`, vendored) | **ACCEPTED** as an implementation detail | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` §2 (DEV-1) |
| CROMA input normalisation | **RESOLVED** 2026-09-18: option (a) adopted; (c) gated experiment; (b) rejected | `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` (DEV-2) |
| CROMA patch count (225) | **CONSISTENT but UNVERIFIED** upstream; checked at load time | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` §4 (DEV-3) |
| `sa_mode: BAM` | **declared, NOT IMPLEMENTED** — raises rather than aliasing PAM | `specialists/change/stanet.py` |
| Change head wired into serving by default | **DEFERRED** — registry override, not config | `specialists/change/specialist.py` |
| Grounding head wired into `inference.py` | **DEFERRED** (Phase 8 integration step) | `docs/PHASE8_GROUNDING_HEAD_DECISION.md` |
| `optical.normalization` / `sar.representation` config keys | **OPEN** — read by no code | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` §6 item 6 |

### 4.1 The grounding resolution experiment — 448 REJECTED

**Rule, fixed before the result was seen** (`docs/PHASE7_RESOLUTION_DECISION.md`; recorded in the
artifact as `rule_changed_since_preregistration: false`):

```
448 WINS  if Recall@0.5 improves by >= 0.05 absolute
          OR mean best IoU improves by >= 0.05 absolute
224 WINS  otherwise
INCONCLUSIVE if fewer than 30 samples were scored
```

**Run:** full VRSBench eval split, **16,159 / 16,159 records scored at both resolutions**, Tesla T4,
`--all --device cuda --tag full`.

**Result: 224 WINS.**

```
recall@0.5 gain 448/224 : -0.0022
bestIoU    gain 448/224 : -0.0147
latency          ratio  : 1.59x
```

Neither component came close to the +0.05 margin. Both were **negative**.

| Metric | 224 | 448 | delta |
|---|---|---|---|
| token grid | 7 × 7 = 49 | 14 × 14 = 196 | 4.0× tokens |
| attention cost (n²) | 1× | 16× | — |
| with boxes | 16159/16159 | 16159/16159 | — |
| **mean best IoU** | **0.0972** | 0.0825 | **−0.0147** |
| Recall@0.10 | **0.3298** | 0.2599 | **−0.0699** |
| Recall@0.25 | **0.1187** | 0.0944 | **−0.0243** |
| Recall@0.50 | **0.0234** | 0.0212 | **−0.0022** |
| matched IoU | 0.0972 | 0.0825 | −0.0147 |
| latency mean | **20.0 ms** | 31.8 ms | **1.59×** |
| latency p90 | **20.9 ms** | 32.9 ms | 1.57× |
| peak VRAM | **592.1 MB** | 599.8 MB | +7.7 MB |
| wall time | **~8.5 min** | ~11.2 min | 1.32× |

**448 is worse on every quality metric and slower. There is no axis on which it wins.**

**Best-IoU distribution** — a whole-distribution move toward the zero-overlap bucket, not a tail
effect:

| bucket | 224 | 448 |
|---|---|---|
| 0.00–0.10 | 10,829 | 11,957 |
| 0.10–0.25 | 3,412 | 2,677 |
| 0.25–0.50 | 1,540 | 1,182 |
| 0.50–0.75 | 336 | 307 |
| 0.75–1.01 | 42 | 36 |

**Paired analysis — independent confirmation.** Both resolutions scored the **same 16,159 samples**, so
the paired test is the stronger statistic: it removes between-object variance.

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

**The pre-registered rule and the paired test AGREE on 224.** There is no rule-versus-evidence
disagreement to escalate: both say 224, and the confidence interval excludes zero by a wide margin. The
win/loss split is also informative — 448 wins on only 8.5 % of records and loses on 20.9 %; the finer
grid is not merely neutral, it is **actively harmful** on a fifth of the corpus.

**Paired recall ladder** — the gap **narrows as the threshold rises**, the signature of a method that
cannot reach high IoU either way:

| threshold | 224 | 448 | diff | 95% CI |
|---|---|---|---|---|
| 0.10 | 0.3298 | 0.2599 | −0.0699 | excludes zero |
| 0.25 | 0.1187 | 0.0944 | −0.0243 | excludes zero |
| 0.50 | 0.0234 | 0.0212 | −0.0022 | excludes zero |

**Why 448 did not help — the honest reading.** The zero-shot method selects a patch by text similarity
and returns that patch's box. At 224 a box is 1/7 of the image; at 448 it is 1/14. A finer grid is only
better if the target is small **and** the similarity peak lands on the correct fine cell. Two things
work against that here: (1) the peak is not sharper at 448 — splitting each cell into four gives four
chances to pick a wrong sub-cell, and the similarity field on frozen features is smooth, so the argmax
moves around (losses outnumber wins 2.5 : 1); (2) **Recall@0.10 drops the most (−0.0699)** — if finer
tokens genuinely localised better, the *loosest* threshold would benefit most. It degrades most, which
means the fine grid adds positional noise rather than positional precision. **This is the zero-shot
baseline's limitation, not a property of RemoteCLIP.**

**Degeneracy notes: none.** At n=12, n=40 and n=6 the smoke runs reported `Recall@0.5 = 0.0000` at
**both** resolutions and emitted a degeneracy warning. At full scale the metric is non-zero
(0.0234 / 0.0212), so the note correctly did not fire — a genuine sub-floor artefact that the full run
resolved, which is why the sub-floor runs were never treated as evidence.

**What this establishes:** grounding runs at 224, frozen in `configs/base.yaml`
(`grounding.image_size: 224`, `grounding.resolution_frozen: true`); peak VRAM for the frozen encoder at
224 is 592 MB (inside the ZeroGPU free tier and a T4); encoder latency at 224 on a T4 is 20 ms/image;
the 224 localisation floor is 1/7 of image width per token.

**What this does NOT establish:** whether the zero-shot baseline is good (it is not — 0.0972 / 0.0234
are weak, an ablation floor for the Phase 8 head); whether a **trained** head has the same resolution
sensitivity (re-opening the question after Phase 8 is legitimate *if* the head's validation curve
suggests it, and would be a new pre-registered experiment, not a silent retune); anything about hidden
ISRO/SAC imagery.

---

## 5. Calibration — a measured negative result

**Status:** `MEASURED`; the calibration is **not an improvement**.

Temperature scaling is enabled (`confidence.temperature_scaling: true`) with
`artifacts/calibration_v001.json` (`schema: calibration_v1`). It is scoped to the **R-02 change-VQA
head's answer confidence only** — *other specialists emit their own raw scores and are unaffected*
(`scope.note`).

| Field | Value | Exact key path |
|---|---|---|
| **temperature** | **0.9772731820958189** (≈ 0.9772732) | `temperature_scaling.temperature` |
| fitted on | `Val` | `temperature_scaling.fitted_on` |
| n samples | 16,441 | `temperature_scaling.n_samples` |
| method | `temperature_scaling` | `provenance.method` |
| objective | `mean_negative_log_likelihood` | `provenance.objective` |
| optimizer | `golden_section_on_log_temperature` | `provenance.optimizer` |
| space | `multiclass_logits` | `provenance.space` |
| iterations | 200 | `provenance.iterations` |
| `log_temperature` | −0.022989052824434128 | `fit_diagnostics.log_temperature` |
| `hit_bound` | false | `fit_diagnostics.hit_bound` |
| `effective` | true | `fit_diagnostics.effective` |
| held-out splits excluded | `[Test, Test2]` | `provenance.held_out_splits_excluded` |
| checkpoint sha256 | `cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a` | `provenance.checkpoint_sha256` |

**The result — ECE got worse:**

| Metric | Before | After | Improvement |
|---|---|---|---|
| **ECE** | **0.013755** | **0.014929** | **−0.001174** |
| NLL | 0.689741 | 0.689631 | +0.00011 |

`ece_improvement = −0.001174` is **negative**: expected calibration error **increased**. NLL improved
by a negligible 0.00011 (fit diagnostics record 0.0001103574982127542). `n_bins: 15`,
`n_classes: 19`, `n_samples: 16441`.

**The scaling is retained because it is part of the frozen configuration, not because it helped.** This
is recorded as a negative result, not smoothed over.

**The reliability diagram is explicitly labelled pre-scaling.** `artifacts/calibration_v001.json` →
`reliability_diagram.ece` is **0.013755** (the *before* value), and the block carries the note: *"Equal-
width bins over predicted-class confidence. ECE is bin-count sensitive and is not an aggregate score."*
The diagram on the Benchmark page is labelled **pre-scaling** so a reader cannot mistake it for the
calibrated result. The bin populations range from 8 (bin 0.133–0.2) to 2,814 (bin 0.933–1.0), and two
bins (0.0–0.0667 and 0.0667–0.1333) are empty.

**The consumer contract** (`consumer_contract`): class `TemperatureCalibration` in module
`evidence.confidence`; applied as `sigmoid(logit(z) / T)` for a scalar `z` and `softmax(logits / T)`
for a distribution; read keys `temperature`, `fitted_on|split`, `artifact`, `n_samples`; resolved by
`load_calibration(config, base_dir='configs')`.

**Limitations.** ECE is bin-count sensitive and is not an aggregate score. The calibrated curve is not
plotted — only the pre-scaling diagram is. The temperature is fitted only on the change-VQA head; no
other specialist is calibrated.

---

## 6. Distribution and licensing

### 6.1 Backbones are not redistributed

Every backbone is fetched from the Hugging Face Hub at run time, pinned by revision (§2). No backbone
weight is included in this release, and no backbone is fine-tuned. Their licences are their own — see
each model's HF page:

| Backbone | Repository | Pinned revision |
|---|---|---|
| Router encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` |
| VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` |
| Grounding | `chendelong/RemoteCLIP` | `bf1d8a3ccf2d` |
| Optical-SAR | `antofuller/CROMA` | `0dd28e3d633b` |

The change encoder's ResNet-18 trunk is a torchvision model with ImageNet weights; it is likewise not
redistributed.

### 6.2 The six trained artifacts are published by this project

The six trained artifacts (§1) are published on the Hugging Face Hub under `thundercode/SatQuery`
(`models/manifest.json` → `release_repo`), **labelled by kind**, each with its backbone dependency
documented and each accompanied by a checksum:

- [`../models/manifest.json`](../models/manifest.json) — schema `satquery_model_manifest_v1`,
  generated by `release/tools/generate_model_manifest.py`, `artifact_count: 6`,
  `generated_utc: 2026-09-25T18:15:38+00:00`, `config_hash: 78f1e3700da15aa1`. Its `note` states: *"No
  byte count or hash is typed by hand."*
- [`../models/checksums.sha256`](../models/checksums.sha256) — six lines, keyed by the path in the
  repository, verifiable with `sha256sum -c checksums.sha256`.

See [`../HF_RELEASE_VERIFICATION.md`](../HF_RELEASE_VERIFICATION.md) for the upload/verification record.

### 6.3 No licence file exists — OPEN

**There is no `LICENSE` file in the source repository.** This is an **OPEN** item flagged in
[`LIMITATIONS.md`](LIMITATIONS.md) §5 (item 29): a licence must be selected **before any public release
of the code**. Model weights carry the terms of their backbone licences; the six trained artifacts are
distributed under whatever terms the owner selects.

### 6.4 What this release does not contain

- **No backbone weights** (fetched at run time, pinned by revision).
- **No training checkpoints** (archived as provenance, not released — §1.2).
- **No feature caches** (`fusion_features/`, the grounding feature cache, etc. — reproducible, not
  released as weights).
- **No hidden/private evaluation data** (`evaluation.hidden_data_access: false`).

---

## 7. Status summary — what is NOT established

### 7.1 Status of each artifact

| Artifact | Implemented | Measured | Acceptance | Ruling |
|---|---|---|---|---|
| `change` | yes | yes | **VERIFIED** | closed |
| `change_vqa` | yes | yes | MEASURED | **OPEN** |
| `optical_sar` | yes | yes | MEASURED | **OPEN** |
| `grounding` | yes | yes | MEASURED (2 protocols) | Phase 8 complete; head not wired |
| `router` | yes | validation only | MEASURED | **TEST NOT RUN** |
| `vlm` | yes | yes | **ACCEPTANCE-REJECTED** | CLOSED |

### 7.2 NOT RUN / OPEN / BLOCKED for this topic

| Item | State |
|---|---|
| **System-level end-to-end accuracy** | **NOT RUN — none exists**; no such number is claimed anywhere |
| **Router test split** | **NOT RUN** |
| **Grounding resolution at 448 for the trained head** | **NOT RUN** — only the zero-shot baseline was measured at 448; a re-open would be a new pre-registered experiment |
| **`grounding_head` wired into `inference.py`** | **DEFERRED** (Phase 8 integration step) |
| **`change` head wired into serving by default** | **DEFERRED** (registry override, not config) |
| **`change_vqa` serving** | requires both a trained head *and* a trained feature extractor; neither wired by default |
| **`sa_mode: BAM`** | **NOT IMPLEMENTED** (raises) |
| **Router adapter parameter count** | **UNKNOWN** — 50,822 (manifest/docstring) vs 51,725 (Phase 4 report), unreconciled |
| **`optical_sar` / `change` head parameter counts** | `UNKNOWN — not established from the available evidence` (manifest records `null`) |
| **`optical.normalization` / `sar.representation` config keys** | read by **no code** — **OPEN** |
| **CROMA `number_of_patches`** | **UNVERIFIED** upstream; checked against 225 at load time |
| **Grounding validation IoU** | the Phase 8 table records 0.0943 at epoch 20; the prose of the same document rounds it to 0.0946 — the small discrepancy is unreconciled |
| **BigEarthNet multi-label evaluation** | **NOT PRODUCED** — the local subset is single-label |
| **Cross-dataset generalisation** | **NOT RUN** — each specialist is evaluated only on its own training-family test split |
| **`LICENSE` file** | **OPEN** — none exists |
| **B-07 tunnel gaps** | patch prepared, **NOT deployed**. **OPEN** |
| **B-02 `codespace_name` trailing `\n`** | cosmetic; **OPEN** |

### 7.3 Explicit non-claims

- **No claim of state-of-the-art performance** on any benchmark.
- **No claim that the trained heads generalise** beyond their training-family test splits.
- **No claim that calibration improves confidence** — the measured ECE worsened.
- **No claim that the VLM adapter is accepted** for production use.
- **No claim of a system-level accuracy**, because no system-level benchmark exists.
- **No claim that any artifact's accuracy may be quoted without its companion metric** (optical-SAR
  accuracy without macro-F1; grounding under one protocol; change-VQA on one test set).

---

## 8. Evidence index

| Topic | Evidence |
|---|---|
| Manifest + checksums | [`../models/manifest.json`](../models/manifest.json), [`../models/checksums.sha256`](../models/checksums.sha256), `release/tools/generate_model_manifest.py`, `release/tools/model_manifest_report.txt` |
| Model card | [`../MODEL_CARD.md`](../MODEL_CARD.md) |
| HF release verification | [`../HF_RELEASE_VERIFICATION.md`](../HF_RELEASE_VERIFICATION.md) |
| Config registry | `configs/base.yaml`, `core/config.py` |
| `change` | `specialists/change/stanet.py`, `specialists/change/specialist.py`, `artifacts/change/eval_test/eval_result.json` |
| `change_vqa` | `specialists/change/vqa_specialist.py`, `artifacts/change_vqa/run/PROMOTION.json` |
| `optical_sar` | `specialists/optical_sar/{croma,fusion_head,specialist}.py`, `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json`, `docs/PHASE12_115_METRIC_COMPUTED.md`, `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| `grounding` | `specialists/grounding/{head,remoteclip}.py`, `artifacts/grounding/remoteclip_grounding_v001/eval_result_{canonical,matched6}.json`, `docs/PHASE7_RESOLUTION_DECISION.md`, `docs/PHASE8_GROUNDING_HEAD_DECISION.md` |
| `router` | `router/adapter.py`, `artifacts/router/threshold_sweep_val.json`, `docs/PHASE4_ROUTER_REPORT.md` |
| `vlm` | `specialists/vqa/model.py`, `artifacts/vlm/phase6_closure.json`, `artifacts/vlm/run1_test_recovery/adapter_verification.json`, `docs/PHASE6_CLOSURE.md` |
| Calibration | `artifacts/calibration_v001.json` |
| Cross-links | [`BENCHMARKS.md`](BENCHMARKS.md) (metrics and their rules), [`EVALUATION.md`](EVALUATION.md) (how each number was produced), [`TRAINING.md`](TRAINING.md) (how each head was trained), [`DATASETS.md`](DATASETS.md) (the corpora), [`LIMITATIONS.md`](LIMITATIONS.md) (the honest catalogue), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (how to reproduce) |
