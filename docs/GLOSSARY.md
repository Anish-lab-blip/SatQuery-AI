# Glossary — deep reference

**Purpose.** This is the vocabulary the rest of the documentation uses, defined precisely, with the
file that **defines or enforces** each term. A term is only listed if it appears in the repository; a
definition is only given if a file supports it.

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` ·
`ATTEMPTED` · `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

**The single most important rule in this document: do not fabricate.** Every definition is grounded in
a file that was read, and the file is named. Where a term's meaning is not established, this document
writes `UNKNOWN — not established from the available evidence` rather than inventing one.

---

## How to use this glossary

The entries are grouped so that a reader can read one group at a time:

| Part | Group | Jump |
|---|---|---|
| A | Tasks | [§A](#part-a--tasks) |
| B | Models and methods | [§B](#part-b--models-and-methods) |
| C | Metrics and losses | [§C](#part-c--metrics-and-losses) |
| D | Data and geospatial | [§D](#part-d--data-and-geospatial) |
| E | System concepts | [§E](#part-e--system-concepts) |
| F | Deployment and operations | [§F](#part-f--deployment-and-operations) |
| G | Status vocabulary | [§G](#part-g--status-vocabulary) |
| H | Index — where each term is defined or enforced | [§H](#part-h--index-where-each-term-is-defined-or-enforced) |

Each entry gives a **precise definition**, the **file that defines or enforces it**, and, where
relevant, a **cross-link** to the deep chapter that covers it in full.

---

## Part A — Tasks

The task vocabulary is a **closed set** with exactly **seven** members, defined once in
`core/schemas.py` (`class Task(str, Enum)`) and mirrored in the router's label space
(`router/label_space.py::TASK_CLASSES`). The three are not interchangeable and the planner must not
substitute one for another — `core/schemas.py:41-46` says so in a comment on `CHANGE_VQA`.

| Term | Definition | Defined / enforced |
|---|---|---|
| **`vqa`** | Visual question answering about **one** asset: a natural-language question in, a short answer out. Runs through the SmolVLM specialist. | `core/schemas.py:36` (`Task.VQA`); `router/label_space.py`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §7 |
| **`caption`** | A prose description of **one** asset. Same specialist as `vqa`, a different prompt kind. | `core/schemas.py:37`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §10 |
| **`grounding`** | Localisation: a text phrase in, boxes out. RemoteCLIP encoder + a trained head. The VLM is **never** the source of coordinates (finding C-5). | `core/schemas.py:38`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §11–§17 |
| **`change`** | The change **detector**: two temporally corresponding assets in, a spatial change map out, **no language output**. STANet-style Siamese network. | `core/schemas.py:39`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §18–§26 |
| **`change_vqa`** | **R-02.** Two temporally corresponding assets **plus** a change-oriented question in, a short answer out. Distinct from `change` (which returns a map, no language) and from `vqa` (which answers about one asset). | `core/schemas.py:41-46`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §27–§32 |
| **`optical_sar`** | Fusion of an optical and a SAR asset: CROMA encoder + a trained fusion head over 19 classes. Requires **two** modalities. | `core/schemas.py:40`; `configs/base.yaml` (`croma`, `fusion`); [`architecture/05-specialists.md`](architecture/05-specialists.md) §33+ |
| **`unsupported`** | The **refusal** class. Returned when nothing matches — never a guess. The lexical fallback returns `unsupported` with low confidence rather than inventing a task (`router/fallback.py`). | `core/schemas.py:47`; `router/fallback.py`; enforced by `core/config.py:198-206` (the task list **must** include `unsupported`) |

**Related task facts.**

- **`temporal` / `spatial_output` / `language_output`** are three binary heads, not tasks.
  `TEMPORAL_TASKS = {"change"}`, `SPATIAL_TASKS = {"grounding"}`, `DUAL_MODALITY_TASKS = {"optical_sar"}`
  (`router/label_space.py`). `Intent._consistency` (`core/schemas.py:107-114`) forces `temporal = True`
  for `change`/`change_vqa` and `modality = optical_sar` for `optical_sar`.
- **Asset-count contract.** `change`, `change_vqa` and `optical_sar` require **exactly 2** assets; `vqa`
  and `caption` require **exactly 1** ([`MODELS.md`](MODELS.md) §3.0.3). The authoritative source is
  `core/planner.CAPABILITY_ASSETS` and `SpecialistSpec.requires_assets`
  (`docs/DEPLOYMENT_ARCHITECTURE.md` §2.2).

---

## Part B — Models and methods

### B.1 Backbones (frozen)

| Term | Definition | Defined / enforced |
|---|---|---|
| **RemoteCLIP** | The frozen vision-language encoder used by `grounding`. Repository `chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt`, revision `bf1d8a3ccf2d`. A CLIP ViT-B/32 adapted to remote sensing. **Measured contract:** patch size 32, transformer width **768**, **projected** dim **512** (`visual.proj` is `(768, 512)`), 151,277,313 parameters. At 224 px it yields a 7×7 = 49-token grid. | `configs/base.yaml` (`grounding`); `specialists/grounding/remoteclip.py` (`_verify_contract`); [`MODELS.md`](MODELS.md) §2.3 |
| **CROMA** | The frozen multi-modal encoder used by `optical_sar`. Repository `antofuller/CROMA`, file `CROMA_base.pt`, revision `0dd28e3d633b`. `encoder_dim` 768, `image_resolution` 120 (15×15 = 225 patches), `optical_channels` 12, `sar_channels` 2. **Asymmetric** (`s1_depth=6`, `s2_depth=12`), with **directional** joint cross-attention (SAR queries optical). 194,365,440 parameters. | `configs/base.yaml` (`croma`); `specialists/optical_sar/`; [`MODELS.md`](MODELS.md) §2.4 |
| **SmolVLM** | The frozen vision-language model used by `vqa` and `caption`. `HuggingFaceTB/SmolVLM-500M-Instruct`, revision `a7da5b986cb5`, ~1015 MB safetensors, 516,165,824 base parameters. | `configs/base.yaml` (`vlm`); [`MODELS.md`](MODELS.md) §2.2 |
| **MiniLM** | The frozen sentence encoder under the intent router. `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4`, 90.9 MB, 22,713,216 parameters, **384**-dim embeddings. Tokenizer ceiling **256**; truncation set to **128**. | `configs/base.yaml` (`router`); `router/encoder.py`; [`architecture/04-router.md`](architecture/04-router.md) §2 |
| **STANet** | The architecture family the `change` detector follows: a **Siamese** network with **spatial-temporal attention** over feature differences and a progressive decoder. Reimplemented rather than vendored (the upstream is Python 3.6-era and depends on `visdom`/`apex`). Weights are **tied**, not copied: both branches call the same `SharedResNetEncoder` instance. | `specialists/change/stanet.py`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §18 |

### B.2 Attention modules

| Term | Definition | Defined / enforced |
|---|---|---|
| **PAM** | The **spatial attention** module used in the change detector (STANet's spatial path). `SpatialAttention` is PAM-style: 1×1 `query`/`key` convolutions to `hidden = max(1, channels // reduction)` with `reduction = 8`, a 1×1 `value`, a 1×1 `out`, `softmax(q @ k / sqrt(hidden))` over positions, and a residual. `configs/base.yaml` sets `change.sa_mode: PAM`. | `specialists/change/stanet.py`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §18.4 |
| **BAM** | The alternative attention mode in the same `sa_mode` enum (`BAM | PAM`). **It raises** — the constructor accepts the value only to fail visibly rather than silently aliasing PAM. | `configs/base.yaml` (`change.sa_mode`); `STANetStyleChangeDetector.__init__`/`.forward`; [`MODELS.md`](MODELS.md) §3.0.3 |

### B.3 Adaptation and calibration methods

| Term | Definition | Defined / enforced |
|---|---|---|
| **LoRA** | Low-Rank Adaptation: a small trainable **delta** attached to a frozen model's weight matrices. Here it is applied to the SmolVLM **text-model projections** with `r=16`, `alpha=32`, `dropout=0.05`. 8,683,520 trainable parameters on a 516,165,824-parameter base (1.68%). | `configs/base.yaml` (`training.lora_*`); `artifacts/vlm/phase6_closure.json`; [`MODELS.md`](MODELS.md) §3.6 |
| **PEFT** | Parameter-Efficient Fine-Tuning — the library/technique family that provides the LoRA adapter. The adapter is loaded with `peft.PeftModel.from_pretrained(model, dir)`. | `specialists/vqa/model.py::_attach_adapter`; [`MODELS.md`](MODELS.md) §1.1 |
| **Temperature scaling** | A **post-hoc** calibration that divides logits by a single fitted scalar `T` before the sigmoid, to bring predicted probabilities closer to observed frequencies. Here `T = 0.9772731820958189`, fitted on `Val` with `n_samples = 16441`. | `evidence/engine.py` (`calibrate`, `confidence_for`); `artifacts/calibration_v001.json`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §8 |
| **`_is_effective`** | The rule that a "fitted" `T` of exactly 1.0 is reported as **uncalibrated** — a `T` of 1.0 applies no transform, so claiming calibration would be a lie. | `evidence/engine.py`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §8.3 |

> **The calibration result is a measured negative.** ECE went **0.013755 → 0.014929** (worse). The
> calibration path is retained **only because it is in the frozen config**, not because it helped
> (`DOCS_STYLE_GUIDE.md` §3; [`LIMITATIONS.md`](LIMITATIONS.md) item 7). The path **is** live
> (`core/controller.py:218` loads it; `evidence/engine.py:643` applies it).

---

## Part C — Metrics and losses

Metric implementations live in the evaluation layer; the honesty rules that govern how they are
reported are in [`EVALUATION.md`](EVALUATION.md) §2 (the eight rules) and
[`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md).

| Term | Definition | Where it is used / defined |
|---|---|---|
| **IoU** | Intersection over Union — `|A ∩ B| / |A ∪ B|`. For grounding, the overlap between a predicted box and a ground-truth box; the headline is **mean best IoU**. For change, the pixel-level overlap of predicted and true change masks. | `docs/PHASE7_RESOLUTION_DECISION.md`; `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json`; `artifacts/change/eval_test/eval_result.json` |
| **mIoU** | Mean IoU across classes — the **macro** IoU. For binary change this is the mean of the two class IoUs (change / no-change). | `artifacts/change/eval_test/eval_result.json` → `metrics.macro.miou` = 0.8457, `metrics.pooled.miou` = 0.9007 |
| **Macro vs pooled** | **Pooled** aggregates all pixels (or all items) into one confusion matrix before computing the metric; **macro** computes the metric per class and averages, giving every class equal weight regardless of frequency. On an imbalanced corpus the two differ sharply, and both must be reported — never one alone. | [`DATASETS.md`](DATASETS.md) §3.5; [`EVALUATION.md`](EVALUATION.md) Rule 4 |
| **F1** | Harmonic mean of precision and recall. Reported both pooled (0.8964) and macro (0.7962) for change. | `artifacts/change/eval_test/eval_result.json` |
| **`exact_match`** | The fraction of VLM answers that match the reference answer **exactly** after normalisation. The VLM adapter's primary endpoint: **0.963** on 1,000 adapted-test questions. | [`architecture/05-specialists.md`](architecture/05-specialists.md) §9.3 |
| **`recall@k`** | The fraction of items where a correct prediction is found within the top `k` (here a threshold ladder: `Recall@0.10`, `@0.25`, `@0.50` for grounding IoU). | `artifacts/grounding/.../eval_result_canonical.json` → `results.head_threshold.recall`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §16.5 |
| **BCE** | Binary cross-entropy. Used for the router's three binary heads (via `BCEWithLogitsLoss`) and for the change detector's change/no-change logit. The **logits form** is mandatory — see below. | `router/label_space.py`; `specialists/change/`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §29 |
| **Dice** | A region-overlap loss closely related to IoU, used alongside BCE for change (`bce_weight: 0.5`, `dice_weight: 0.5`). *"Why Dice is not decoration"* — it counteracts the class imbalance BCE alone handles poorly. | `configs/base.yaml` (`change.bce_weight`, `change.dice_weight`); [`architecture/05-specialists.md`](architecture/05-specialists.md) §20.1 |
| **GIoU** | Generalised IoU — IoU extended with a penalty for non-overlapping boxes, giving a gradient even when boxes do not intersect. The grounding head's box loss uses `box_loss_weight: 0.5`, `giou_loss_weight: 0.3`, `confidence_loss_weight: 0.2`. | `configs/base.yaml` (`grounding_training`); [`architecture/05-specialists.md`](architecture/05-specialists.md) §13.6 |
| **NMS** | Non-Maximum Suppression — merges overlapping candidate boxes, keeping the highest-scoring and suppressing the rest above an IoU threshold. Grounding uses `nms_iou: 0.50`, `max_candidates: 20`, implemented in pure torch. | `configs/base.yaml` (`grounding.nms_iou`); `specialists/grounding/`; [`architecture/05-specialists.md`](architecture/05-specialists.md) §13.5 |
| **ECE** | Expected Calibration Error — the average gap between predicted confidence and observed accuracy across confidence bins. Here it went **0.013755 → 0.014929** after temperature scaling (worse). | `artifacts/calibration_v001.json`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §8; [`LIMITATIONS.md`](LIMITATIONS.md) item 7 |
| **NLL** | Negative log-likelihood — a proper scoring rule; improved marginally under calibration even though ECE worsened. | `artifacts/calibration_v001.json`; `docs/API_CONTRACT.md` §4 |

> **The saturated-BCE defect.** A `BCE` implemented from `sigmoid`-then-`BCE` divides the backward pass
> by `p(1−p)`, which explodes as `p → 0` or `1`. `BCEWithLogits` **cannot** saturate. The change-VQA
> head therefore uses the logits form, and the fallback is constrained accordingly
> ([`architecture/05-specialists.md`](architecture/05-specialists.md) §29).

---

## Part D — Data and geospatial

Dataset details are in [`DATASETS.md`](DATASETS.md); leakage controls are in
[`EVALUATION.md`](EVALUATION.md) §3.2–§3.3 and `configs/base.yaml` (`evaluation.leakage_split_key: scene_id`).

### D.1 Datasets

| Term | Definition | Defined / used |
|---|---|---|
| **LEVIR-CD-256** | The change-detection corpus: 256×256 bitemporal building-change tiles. Split `train: 7120`, `val: 1024`, `test: 2048` (`configs/base.yaml` → `change.levir_split`). The test split is the **only** headline that carries the `VERIFIED` tag. | [`DATASETS.md`](DATASETS.md) §3; `artifacts/change/eval_test/eval_result.json` |
| **VRSBench** | The grounding corpus. Boxes are stored **normalised to 0–100**; the project stores 0–1, hence `grounding.benchmark_box_scale: 100.0`. Train split has **noisy ground truths** (recorded). The eval used **16,159** records. | [`DATASETS.md`](DATASETS.md) §5; `configs/base.yaml` (`grounding`) |
| **CDVQA** | One of the two change-VQA corpora (question/answer annotations). | [`DATASETS.md`](DATASETS.md) §4 |
| **SECOND** | The other change-VQA corpus. **Measured finding:** *Test and Test2 are the same 968 images*; and `change_ratio_types` has a **different vocabulary in every split**. The imagery is **not shipped** with the annotations. | [`DATASETS.md`](DATASETS.md) §4.2–§4.7 |
| **BigEarthNet** | The optical-SAR fusion corpus (the reBEN v2 corpus) and a separate VLM-adaptation subset. **19 CLC classes.** The local subset is **100 % single-label**, unlike the official 1–11 multi-label scheme, so its metrics are **not comparable** to published numbers. | [`DATASETS.md`](DATASETS.md) §6; `configs/base.yaml` (`fusion.num_classes: 19`) |
| **CLC classes** | The **CORINE Land Cover** class taxonomy — the 19 label classes the fusion head predicts. **Measured limitation:** the live service returns `class_18`, a bare class index, **not** a human-readable CLC label. | `configs/base.yaml` (`fusion.num_classes: 19`); [`LIMITATIONS.md`](LIMITATIONS.md) item 6 |
| **Single-label vs multi-label** | A sample is **single-label** if it carries exactly one class; **multi-label** if it carries several. BigEarthNet's official scheme is multi-label (1–11); the project's local subset is single-label, which is why the fusion metrics must not be compared to published BigEarthNet numbers. | [`DATASETS.md`](DATASETS.md) §6.4; [`LIMITATIONS.md`](LIMITATIONS.md) item 21 |

### D.2 Geospatial and raster terms

| Term | Definition | Defined / enforced |
|---|---|---|
| **GeoTIFF** | A TIFF raster carrying georeferencing (a CRS, an affine transform, bounds). The project reads rasters via `preprocessing/raster.py` and records geo metadata in `GeoMetadata` (`core/schemas.py:120`). | `preprocessing/raster.py`; `core/schemas.py` (`GeoMetadata`) |
| **Band** | One channel of a raster (e.g. a single spectral band). `SensorDescriptor.available_bands` / `band_map` describe which bands a sensor provides. CROMA expects exactly **12** optical and **2** SAR channels. | `core/schemas.py:138` (`SensorDescriptor`); `configs/base.yaml` (`croma.optical_channels: 12`, `sar_channels: 2`) |
| **Optical** | Imagery in the visible/near-infrared; the default modality. CROMA expects **12** optical channels; the project normalises with a percentile stretch. | `configs/base.yaml` (`optical`); `core/schemas.py` (`Modality.OPTICAL`) |
| **SAR** | Synthetic Aperture Radar imagery. CROMA expects **2** SAR channels (VV, VH). Represented in **dB**. | `configs/base.yaml` (`sar`); `core/schemas.py` (`Modality.SAR`) |
| **dB** | Decibels — the logarithmic representation of SAR backscatter. `configs/base.yaml` sets `sar.representation: db`, `clip_min_db: -30`, `clip_max_db: 5`. | `configs/base.yaml` (`sar`) |
| **Percentile stretch** | A deterministic contrast stretch mapping the 2nd and 98th percentiles of finite values to the display range. *"The same file always yields the same"* result. Implemented in `preprocessing/imagery.py`. | `preprocessing/imagery.py:63`; `configs/base.yaml` (`optical.normalization: percentile`, `lower_percentile: 2`, `upper_percentile: 98`) |
| **`scene_id`** | A stable identifier for the acquisition scene a sample came from. It is the **leakage split key**: `evaluation.leakage_split_key: scene_id`. Splits are made by scene, **never by example**. | `configs/base.yaml` (`evaluation`); `core/schemas.py:163` (`AssetMetadata.scene_id`); [`DATASETS.md`](DATASETS.md) §7.3 |
| **Split** | A named partition of a corpus (`train` / `val` / `test`). Recorded on `AssetMetadata.split`. The public test split is **immutable** (`evaluation.immutable_public_test: true`). | `configs/base.yaml` (`evaluation`); `core/schemas.py:165` |
| **Leakage** | Any way that information from a test sample influences training, inflating measured performance. The project enforces five leakage rules in code and splits **by group** (template / hard-negative family), placing hard-negative families in **test** so their accuracy measures generalisation, not memorisation (finding F4-3). | `configs/base.yaml` (`evaluation.leakage_split_key`, `router.training.hard_negatives_to_test: true`); [`DATASETS.md`](DATASETS.md) §7.2; [`EVALUATION.md`](EVALUATION.md) §3.2 |

---

## Part E — System concepts

### E.1 The pipeline's nouns

| Term | Definition | Defined / enforced |
|---|---|---|
| **Router** | The **first** of two stages: it reads the query and emits an `Intent`. It is **advisory only** — *"Output of the learned router. Advisory only — the controller decides."* (`core/schemas.py:95`). Implemented as a frozen MiniLM encoder plus a five-head adapter. | `router/`; `core/schemas.py` (`Intent`); [`architecture/04-router.md`](architecture/04-router.md) |
| **Controller** | The **second** stage and the decision authority. It parses, validates, plans, dispatches specialists, aggregates, verifies and responds — the nine-state spine. It may **override** the router (asset-count-aware dispatch, below). | `core/controller.py`; `core/schemas.py` (`ControllerState`); [`architecture/03-request-lifecycle.md`](architecture/03-request-lifecycle.md) |
| **Specialist** | A module that performs exactly one task and returns exactly one `SpecialistResult`. *"Every specialist returns exactly this. No exceptions."* (`core/schemas.py:326`). Five are registered. | `specialists/`; `core/schemas.py` (`SpecialistResult`); [`architecture/05-specialists.md`](architecture/05-specialists.md) §1 |
| **Evidence** | A single, typed, inspectable record supporting a result. It is the project's answer to *"never pretend it knows more than the evidence supports"*. Every `Evidence` has a `type`, a `source_specialist`, and (if spatial) a `coordinate_system`. | `core/schemas.py:216` (`Evidence`); [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §2 |
| **Evidence engine** | The module that turns many specialists' evidence into one ordered, deduplicated, capped collection, and that applies calibration to confidence. Pipeline: collect → dedup → sort → annotate → renumber → cap. | `evidence/engine.py`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §5 |
| **Confidence breakdown** | The structured confidence object: a `raw` value, an optional `calibrated` value, a `method` (`"uncalibrated"` or `"temperature_scaling"`), a `components` map, and a `degraded` flag with a reason. | `core/schemas.py:259` (`ConfidenceBreakdown`); [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §7.2 |
| **Calibrated vs uncalibrated** | **Calibrated** = a fitted transform was applied (`method: "temperature_scaling"`, `effective: true`). **Uncalibrated** = no transform, or a `T` of exactly 1.0 (`method: "uncalibrated"`). Reporting uncalibrated is **honest, not broken**. | `evidence/engine.py`; `core/schemas.py` (`ConfidenceBreakdown.method`) |
| **`ResultEnvelope`** | The **top-level response object** of `POST /v1/analyze`: a `run_id`, a `SpecialistResult`, and the `ExecutionTrace`, plus a `schema_version`. The route returns it **verbatim**, trace included. | `core/schemas.py:421`; `core/controller.py:394`; [`architecture/08-api-contract.md`](architecture/08-api-contract.md) |
| **`ExecutionTrace`** | The per-run record of what happened: `run_id`, `task`, `query`, `inputs`, `modalities`, `intent`, `validation`, `workflow`, `steps`, `selected_models`, `parameters`, `outputs`, `confidence`, `timings`, `fallbacks`, `errors`, `contradiction`, `config_hash`, `started_at`, `finished_at`. It is **client-facing**. | `core/schemas.py:296`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §3 |
| **`Intent`** | The router's structured output: a `task`, a `modality`, three booleans (`temporal`, `spatial_output`, `language_output`), a `confidence`, and a `source` (`learned` | `lexical_fallback` | `forced`). | `core/schemas.py:94`; [`architecture/04-router.md`](architecture/04-router.md) |
| **`EvidenceType`** | The **closed 11-member** vocabulary of evidence kinds: `image_crop`, `tile`, `bounding_box`, `mask`, `change_map`, `optical_view`, `sar_view`, `joint_feature_region`, `statistic`, `geolocation`, `availability_mask`. | `core/schemas.py:65`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §3 |
| **`CoordinateSystem`** | The frame a coordinate is expressed in. **Exactly three** values: `normalized_0_1`, `pixel`, `geo`. *"Never omit this. A bare box is meaningless without it."* (finding C-5). | `core/schemas.py:57`; [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md) §4.1 |
| **`Region`** | A general spatial result: an optional `box`, an optional `mask_ref`, a `label`, a `score`, and a `coordinate_system`. | `core/schemas.py:189` |
| **`ChangeRegion`** | A change-specific region: a required `box`, `area_pixels`, `mean_probability`, an optional `stability`, and a `coordinate_system`. | `core/schemas.py:202` |
| **Availability mask** | A per-channel boolean vector recording **which bands a sensor actually measured**. It is consumed by the **fusion head**, **not** by CROMA (finding C-1) — because handing a masked autoencoder a mask invites it to reconstruct missing channels, precisely the fabrication the sensor adapter exists to prevent. | `core/schemas.py` (`SensorDescriptor.availability_mask`, `EvidenceType.AVAILABILITY_MASK`); `specialists/optical_sar/fusion_head.py`; [`MODELS.md`](MODELS.md) §3.0.1 |

### E.2 Behavioural concepts

| Term | Definition | Defined / enforced |
|---|---|---|
| **Degraded** | A result flag meaning *"the analysis could not be fully performed"* — e.g. no trained detector, or co-registration too poor to support a spatial claim. It is **not** *"no change was detected"*, which is a **normal, successful** outcome. | `core/schemas.py:343` (`SpecialistResult.degraded`); `core/schemas.py:353-406` (the validator, incl. the F-16c correction); [`architecture/05-specialists.md`](architecture/05-specialists.md) §4.2 |
| **Fallback** | A recorded degradation, named in `ExecutionTrace.fallbacks` as a human-readable string. It records **what degraded**, not what was retried. | `core/schemas.py:314`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §3.7 |
| **Lexical fallback** | The **deterministic, model-free** router path: ordered lexical rules, highest specificity first, first match wins. It has **no model, no embeddings, no randomness**, and it *"never invents capability"* — if nothing matches it returns `unsupported` with low confidence. Precedence: `dual_modality > temporal > spatial > caption > vqa > unsupported`. Used when the learned router is low-confidence, or when the encoder cannot load at all. | `router/fallback.py` (`lexical_route`, `self_check`); `core/schemas.py` (`Intent.source = "lexical_fallback"`) |
| **Dispatch** | The controller's act of choosing **which** specialist to run for a request. Dispatch is **asset-count-aware**: a change-style query with **one** asset reads `change` but **dispatches** `change_vqa`. The reading is asset-count-blind; the dispatch is not. | `core/controller.py`; [`architecture/04-router.md`](architecture/04-router.md); [`LIMITATIONS.md`](LIMITATIONS.md) item 13 |
| **Asset-count-aware** | The property that a decision accounts for **how many assets** were supplied. Exactly 2 assets for `change`/`change_vqa`/`optical_sar`; exactly 1 for `vqa`/`caption`. *"A silent 'just use the first two' produces a confident answer to a question the caller did not ask."* | `core/planner.CAPABILITY_ASSETS`; `SpecialistSpec.requires_assets`; [`MODELS.md`](MODELS.md) §3.0.3 |
| **Preview path** | The frontend's **no-file** demonstration mode. It emits **all eight** production event names — the common summary *"it emits no specialist events"* is a **correction**: what the preview withholds is the **content**, not the event. Its specialist events carry a component **name** and no measurement; the result stages carry explicit emptiness. | `frontend/assets/js/mission.js:550-578`; [`architecture/09-frontend.md`](architecture/09-frontend.md) §5.2 |
| **Trace fill** | The execution-trace progress bar's fill fraction. The measured live value is **94.4444 % (17/18)** — the bar advances to the **current node** rather than to its right edge, so a **fully-completed** run stops at 94.4444 %, not 100 %. | `architecture/06-evidence-and-confidence.md` §13.3; `README.md` (trace-fill row); `DOCS_STYLE_GUIDE.md` §3 |
| **Run id** | The identifier for one analysis run, generated as `run_<12 hex>`. It appears on the `ResultEnvelope` and inside the `ExecutionTrace`. | `core/schemas.py:28` (`_new_id`), `:299` (`ExecutionTrace.run_id`); [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §5.1 |
| **Request id** | A **separate** identifier namespace, `req_<24 hex>`, generated by the gateway and echoed as `X-Request-Id`. **The two id spaces do not join** — there is no trace-context propagation. | `gateway/policy.py`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §5.2, §5.4 |
| **Config hash** | A 16-hex-character stable digest of the whole config registry, computed in `Config.hash` over the JSON-sorted registry. It is recorded in every evaluation run and in every artifact. The frozen value is **`78f1e3700da15aa1`**. | `core/config.py:76-80`; [`architecture/07-configuration-freeze.md`](architecture/07-configuration-freeze.md) |
| **Frozen config** | The config registry whose hash is **`78f1e3700da15aa1`** and which may not move without invalidating the benchmark. *"Frozen"* means validated against the frozen architecture at load time and hashed for reproducibility. | `core/config.py`; `configs/base.yaml`; [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §2 |
| **Invariant** | A property **enforced at load time**, not merely documented, because the failure it prevents is a **silent shape error** — a tensor of the right shape that trains to a worse number. Examples: `fusion.input_dim == 2318`, `grounding_head.feature_dim == 2048`, `croma.image_resolution % 8 == 0`. | `core/config.py` (`_validate`); [`MODELS.md`](MODELS.md) §3.0; [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §2.3 |

---

## Part F — Deployment and operations

Topology, transport and the runbook are in [`DEPLOYMENT.md`](DEPLOYMENT.md),
[`architecture/02-deployment-topology.md`](architecture/02-deployment-topology.md) and
[`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md).

| Term | Definition | Defined / enforced |
|---|---|---|
| **Gateway** | The **Render-hosted** edge service. It owns eight responsibilities — schema validation, size limits, rate limiting, CORS, request IDs, timeouts, secret custody, error translation — and **makes no model decisions**. It proxies **four** routes: `/v1/health`, `/v1/capabilities`, `/v1/analyze`, `/v1/assets`. | `gateway/app.py`, `gateway/policy.py`; `docs/DEPLOYMENT_ARCHITECTURE.md` §2.1 |
| **Orchestrator** | The gateway's role as the component that **starts the Codespace** and **waits** for it, then proxies the request. Also called the hub. | `docs/DEPLOYMENT_ARCHITECTURE.md`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §2.3 |
| **Tunnel** | The **live transport**: an **outbound long-poll** (`POST /tunnel/agent`) from a Codespace agent to the hub. Direction is **inverted** — the Codespace dials out, which is why the design works with a **private** repository. A *forwarded* Codespace port returns **HTTP 302** for a private repo, which is *why* the tunnel exists. | `deploy/codespace/tunnel_agent.py`; `tunnel.py`; [`architecture/02-deployment-topology.md`](architecture/02-deployment-topology.md) §3 |
| **Long-poll** | A request that the server **parks** until work arrives or a timeout fires, rather than returning immediately. The tunnel agent announces itself with a long-poll `POST /tunnel/agent`. | `tunnel.py`; [`architecture/02-deployment-topology.md`](architecture/02-deployment-topology.md) §3 |
| **Wake flow** | The **fallback** path: `ensure_codespace_up()` starts a stopped Codespace via the GitHub API and polls `GET {base}/v1/health` every 2 s (10 s per-poll timeout) until it answers 200, bounded by `wake_timeout_s = 120`. | `apply-test/main.py:417-442`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.2 |
| **Cold start** | The delay before a stopped Codespace (or a sleeping Render instance) can serve. Documented as **"tens of seconds"**; bounded worst case `wake_timeout_s = 120` before a `504`. **No measured distribution exists.** | [`DEPLOYMENT.md`](DEPLOYMENT.md) §8; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.2 |
| **Forward path** | The **fallback** transport that proxies to a forwarded Codespace port directly. For a **private** repo it returns `302`; in `auto` mode a tunnel timeout falls through to it, which is the root shape of **B-07**. | [`DEPLOYMENT.md`](DEPLOYMENT.md) §8.1; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.3 |
| **`tunnel_offline`** | The error code (`503`) meaning **no agent is connected** and `mode == "tunnel"`. The remedy is to start the Codespace. | `apply-test/main.py:308-332`; [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) §7.3 |
| **CORS allowlist** | The explicit list of permitted browser origins, from `SATQUERY_ALLOWED_ORIGINS`. **Never `*`** — an empty or wildcard allowlist is **refused at construction** by `GatewayConfig.__post_init__`. | `gateway/policy.py`; `docs/DEPLOYMENT_ARCHITECTURE.md` §2.1 |
| **Rate limit** | A **per-IP** and global request cap. Default `rate_limit_per_ip = 10` over `rate_limit_window_s = 60.0`. It is a **fairness control, not a protection control** — it does not prevent a bypass, and it exists to protect the GPU budget. | `gateway/policy.py`; `docs/DEPLOYMENT_ARCHITECTURE.md` §5.2 |
| **LFS** | Git **Large File Storage** — the mechanism for storing large binary files (model weights) in a Git repository by pointer. Used for the released artifacts on the Hub. | [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §7; [`MODELS.md`](MODELS.md) |
| **Git Data API** | The GitHub REST API used to **create commits programmatically** instead of `git push`. Chosen because it avoids a local clone/credential flow. | [`DEPLOYMENT.md`](DEPLOYMENT.md) §7.1 |

---

## Part G — Status vocabulary

The project uses a **fixed status vocabulary** on every substantive claim. *"Never upgrade a status:
`NOT RUN` never becomes `PASS`; `OPEN` never becomes `VERIFIED`; `REJECTED` never becomes `ACCEPTED`."*
(`DOCS_STYLE_GUIDE.md` §1.7).

| Tag | Definition |
|---|---|
| **`IMPLEMENTED`** | The code exists. It does **not** imply it was run, or that it produced a good result. |
| **`VERIFIED`** | Measured **and** independently confirmed against a frozen criterion. In this project only the change headline carries it (pooled IoU 0.8122 / macro IoU 0.8457 / pooled F1 0.8964). |
| **`MEASURED`** | A number was produced by running something and recorded in an artifact. It does not imply acceptance. |
| **`ATTEMPTED`** | Work was done but did not complete cleanly or did not resolve the question (e.g. the BigEarthNet format contradiction). |
| **`NOT RUN`** | The work was **not** done. Never to be read as a pass. |
| **`BLOCKED`** | The work cannot be done in the environment or without an external resource. A **fact**, not a decision. |
| **`DEFERRED`** | The work could be done now but was deliberately postponed, with a reason. |
| **`REJECTED`** | The work was done and the answer was **no**. Used for the 448 grounding resolution and the ZeroGPU deployment target. |
| **`OPEN`** | The item is unresolved and **not** closed. Used for B-07, B-02, the missing `LICENSE`, and the 50,822-vs-51,725 comment discrepancy. |
| **`RESOLVED`** | A previously open item was **closed with evidence**. |
| **`CLOSED`** | The item is finished and no longer tracked. |
| **`USABLE_VERIFIED`** | The **VLM adapter's** state: its metrics are usable (`exact_match` 0.963), verified against the run record. It is **not** the same as accepted. |
| **`ACCEPTANCE-REJECTED`** | The artifact passed verification but was **not promoted** for production use. The VLM adapter is `USABLE_VERIFIED` **and** `ACCEPTANCE-REJECTED` — *"Those are different words about different things."* The deployed caption/VQA path uses the **unadapted** model. |
| **`TRAINED_UNVERIFIED`** | A training run produced an artifact that has **not** been evaluated on a held-out split. The change-VQA run record's state, with the note: *"training produces an artifact, not a verified capability. R-02 reaches VERIFIED only after the returned checkpoint has been evaluated on the held-out split."* Superseded by the promotion record but retained on disk. |
| **`OPEN` ruling** | A verdict that is **not settled** — the metric is measured but the acceptance decision is unmade. Used for optical-SAR (accuracy 0.931 with macro-F1 0.434161) and change-VQA (test 0.697626/0.378373; test2 0.651469/0.372309). *"Never accuracy without macro-F1."* |

**Related honesty rules that recur with these tags:**

- **`USABLE ≠ ACCEPTED`** ([`EVALUATION.md`](EVALUATION.md) Rule 7).
- **Two protocols are never collapsed** ([`EVALUATION.md`](EVALUATION.md) Rule 2): grounding is reported
  under canonical (0.2838) **and** matched6 (0.2566).
- **Two test sets are never collapsed** (Rule 3): change-VQA has test **and** test2.
- **Validation is not test** (Rule 5): the router's 0.965116 is **validation**, ungated, n = 86; the test
  split was **NOT RUN**.
- **A negative result stays negative** (Rule 6): calibration made ECE worse and stays reported as worse.

---

## Part H — Index — where each term is defined or enforced

| Term / concept | Primary file(s) |
|---|---|
| Task vocabulary (7 tasks) | `core/schemas.py` (`Task`); `router/label_space.py` (`TASK_CLASSES`) |
| Modality vocabulary | `core/schemas.py` (`Modality`); `router/label_space.py` (`MODALITY_CLASSES`) |
| `Intent`, `EvidenceType`, `CoordinateSystem`, `Region`, `ChangeRegion` | `core/schemas.py` |
| `SpecialistResult`, `ResultEnvelope`, `AnalysisRequest`, `HealthStatus` | `core/schemas.py` |
| `ExecutionTrace`, `TraceStep`, `ModelRef`, `ConfidenceBreakdown`, `Evidence` | `core/schemas.py` |
| Config hash, invariants, device resolution | `core/config.py` |
| Frozen values and budgets | `configs/base.yaml` (frozen `configs/deploy.yaml` is inert) |
| Router encoder / adapter / label space / fallback | `router/encoder.py`, `router/adapter.py`, `router/label_space.py`, `router/fallback.py` |
| Controller, dispatch, nine-state spine | `core/controller.py` |
| Planner, capability asset counts | `core/planner.py` |
| Evidence engine, calibration, confidence | `evidence/engine.py`, `evidence/confidence.py` |
| Specialists | `specialists/vqa/`, `specialists/grounding/`, `specialists/change/`, `specialists/optical_sar/` |
| Input-quality gate, percentile stretch | `preprocessing/quality.py`, `preprocessing/imagery.py`, `preprocessing/raster.py` |
| Gateway policy, rate limit, CORS, error taxonomy | `gateway/policy.py`, `gateway/app.py`, `gateway/assets.py` |
| Deployment report / capability adapter | `app/deployment.py`, `app/space_app.py` |
| Six artifact byte counts and digests | [`../models/manifest.json`](../models/manifest.json), [`../models/checksums.sha256`](../models/checksums.sha256) |
| Status vocabulary | `DOCS_STYLE_GUIDE.md` §2; [`EVALUATION.md`](EVALUATION.md) §2 |

### Cross-references

- [`PERFORMANCE.md`](PERFORMANCE.md) — the footprint, the component timings, the cost traps, and the
  explicit statement that **no end-to-end benchmark exists**.
- [`MODELS.md`](MODELS.md) — the artifact and backbone reference, with enforced invariants.
- [`BENCHMARKS.md`](BENCHMARKS.md) — every measured metric with its artifact key path.
- [`EVALUATION.md`](EVALUATION.md) — the eight honesty rules and the protocols.
- [`DATASETS.md`](DATASETS.md) — the corpora, splits and leakage controls.
- [`TRAINING.md`](TRAINING.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — how the artifacts were made
  and how the claims are re-checked.
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — the live topology, transport and cold start.
- [`LIMITATIONS.md`](LIMITATIONS.md) — the exhaustive catalogue of what is not done or not known.
- [`architecture/05-specialists.md`](architecture/05-specialists.md),
  [`architecture/06-evidence-and-confidence.md`](architecture/06-evidence-and-confidence.md),
  [`architecture/10-observability-and-ops.md`](architecture/10-observability-and-ops.md) — the deep
  chapters for the specialists, the evidence/confidence subsystem, and operations.

---

## Terms deliberately not defined here

The following appear in some drafts but are **not** established by any file read, so this glossary does
not define them: a general "quality score", a "system accuracy", an "end-to-end latency", a
"cost-per-request", or a "throughput". Where a reader needs one of these, the honest answer is
`UNKNOWN — not established from the available evidence`, and the place to look is
[`PERFORMANCE.md`](PERFORMANCE.md) §8 and [`LIMITATIONS.md`](LIMITATIONS.md) §3.
