# SatQuery AI — Master Architecture & Implementation Plan (ORIGINAL, HISTORICAL)

> **STATUS: HISTORICAL / SUPERSEDED IN PARTS.** This is the project's original master plan, preserved
> verbatim as the design record. It is **not** a description of the shipped system. Where it and the
> rest of this documentation disagree, the rest of this documentation wins.
>
> **Known superseded points:**
>
> - It specifies a **Gradio GUI**. The shipped system is a **static frontend** on Cloudflare Pages;
>   `app/space_app.py` serves JSON only and deliberately builds no Gradio interface.
> - It specifies an **HF Space + ZeroGPU + Railway** deployment. The shipped system is
>   **Cloudflare Pages → Render → outbound tunnel → GitHub Codespace**, CPU-first.
> - Its **implementation-order phases** (0–19) and stop/go gates describe the plan, not the record of
>   what was executed. For actual status see [`CHANGELOG.md`](CHANGELOG.md),
>   [`LIMITATIONS.md`](LIMITATIONS.md) and the architecture reference in
>   [`ARCHITECTURE.md`](architecture/README.md).
> - Several sections describe capabilities that were later **rejected or measured differently** — for
>   example the grounding resolution (the plan allows 448; the shipped value is 224, after 448 lost a
>   pre-registered paired test) and the VLM adapter (trained, then **acceptance-rejected**).
>
> It is included because the decisions it records — the frozen backbones, the typed contracts, the
> evidence/confidence split, and the "no LLM-generated confidence" rule — are the decisions the system
> still obeys.

---



# SatQuery AI — Revised Master Architecture & Implementation Plan

# 0. Executive Decision

## 0.1 Final architecture

Build SatQuery AI as a **modular monolith**.

One application.

One Python backend.

One Gradio GUI.

One controller.

Multiple independently callable specialist modules with stable typed interfaces.

```
`                              ┌───────────────────────┐`

`                              │       Gradio GUI      │`

`                              │ Upload / Query / Map  │`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │     Query Processor   │`

`                              │ normalize / sanitize  │`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │   Tiny NLP Intent      │`

`                              │       Router           │`

`                              │ MiniLM + Adapter       │`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │ Deterministic Policy   │`

`                              │ / Workflow Controller  │`

`                              └───────────┬───────────┘`

`                                          │`

`                  ┌───────────────────────┼───────────────────────┐`

`                  │                       │                       │`

`                  ▼                       ▼                       ▼`

`          ┌──────────────┐       ┌──────────────┐       ┌──────────────┐`

`          │ VLM          │       │ Grounding    │       │ Change       │`

`          │ Specialist   │       │ Specialist   │       │ Specialist   │`

`          │ SmolVLM      │       │ RemoteCLIP   │       │ STANet       │`

`          └──────┬───────┘       └──────┬───────┘       └──────┬───────┘`

`                 │                      │                      │`

`                 └──────────────────────┼──────────────────────┘`

`                                        │`

`                                        ▼`

`                              ┌───────────────────────┐`

`                              │ Optical-SAR Specialist│`

`                              │       CROMA            │`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │ Evidence Engine       │`

`                              │ boxes / masks / crops │`

`                              │ maps / modality proof │`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │ Confidence Calibration│`

`                              └───────────┬───────────┘`

`                                          │`

`                                          ▼`

`                              ┌───────────────────────┐`

`                              │ Result Normalizer     │`

`                              │ JSON / Answer / Trace │`

`                              └───────────┬───────────┘`

`                                          │`

`                          ┌───────────────┴───────────────┐`

`                          ▼                               ▼`

`                   GUI visualization              PDF/JSON report`
```


## 0.2 Specialist stack

| **Role** | **Model/component** | **Decision** |
| :-: | :-: | :-: |
| Intent understanding | `sentence-transformers/all-MiniLM-L6-v2` + small classifier adapter | **Primary** |
| Main VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | **Primary** |
| Grounding | RemoteCLIP ViT-B/32 + lightweight grounding head | **Primary initial grounding design** |
| Change detection | STANet-style Siamese network | **Primary** |
| Optical-SAR representation | CROMA-base | **Primary** |
| Geospatial processing | Rasterio + pyproj | **Primary** |
| Report generation | ReportLab | **Primary** |
| GUI | Gradio | **Primary** |

RemoteCLIP provides official pretrained RN50, ViT-B/32 and ViT-L/14 checkpoints and is explicitly built as a remote-sensing vision-language model. The official repository provides the `ViT-B-32` checkpoint and OpenCLIP loading path. ([GitHub](https://github.com/ChenDelong1999/RemoteCLIP?utm_source=chatgpt.com))

CROMA's official implementation provides pretrained `base` and `large` weights and explicit optical, SAR and joint representations; its base implementation uses a 768-dimensional representation with separate Sentinel-1 two-channel and Sentinel-2 twelve-channel inputs. ([GitHub](https://github.com/antofuller/CROMA/blob/main/README.md?utm_source=chatgpt.com))


## 0.3 Mandatory capabilities

SatQuery must expose:

```
`Single-image VQA`

`Single-image captioning`

`Text-guided grounding`

`Bi-temporal change analysis`

`Optical-SAR analysis`

`Natural-language query routing`

`Evidence generation`

`Confidence`

`Execution trace`

`GeoTIFF handling`

`Downloadable result/report`
```

The uploaded project contract already requires VQA, an additional single-image task, temporal change, optical-SAR joint analysis and agentic orchestration. Grounding is now added as a first-class capability.


## 0.4 The key architectural principle

The NLP router **understands**.

The policy engine **decides**.

The specialists **compute**.

The VLM **explains**.

The evidence engine **proves**.

That separation is the foundation of the entire system.


# 1. Requirement-to-Implementation Traceability Matrix

| **Requirement** | Mandatory | **Component** | **Dataset** | **Model** | **Training** | **Runtime** | **Evidence** | **Public Test** | **ISRO/SAC** |
| :-: | -: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Single optical/MS image | Yes | Raster loader + VLM | VRSBench / RSVQA | SmolVLM | PEFT | VQA | selected image/tile | Yes | Yes |
| Single SAR | Yes | SAR adapter + VLM | BigEarthNet S1 | SmolVLM/CROMA | adapter | VQA | SAR view | Where relevant | Yes |
| Optical-SAR pair | Yes | Fusion specialist | BigEarthNet S1/S2 | CROMA | fusion head | fusion DAG | optical + SAR + joint | Where prescribed | **Primary** |
| Bi-temporal pair | Yes | Change specialist | LEVIR-CD/CDVQA | STANet-style | supervised | change DAG | change map | Yes | Yes |
| VQA | Yes | VLM | RSVQA/VRSBench | SmolVLM | LoRA | VQA | image/tile | Yes | Yes |
| Captioning | Yes | VLM | VRSBench | SmolVLM | LoRA | caption | image | Yes | Yes |
| Grounding | Added | Grounding specialist | VRSBench referring expressions | RemoteCLIP + head | head/adapter | grounding | boxes/regions | Yes | Yes where spatial output |
| Agentic orchestration | Yes | NLP router + policy engine | synthetic routing set | MiniLM | classifier head | controller | trace | Functional | Yes |
| Evidence | Yes | Evidence engine | all | none | none | postprocessing | boxes/masks/maps/crops | Functional | Yes |
| Confidence | Yes | Calibration module | validation | specialist scores | calibration only | postprocessing | calibrated score | Yes | Yes |
| Execution trace | Yes | Trace builder | all | none | none | controller | structured trace | Yes | Yes |
| Reports | Yes | ReportLab | all | none | none | finalization | PDF/JSON | Functional | Functional |
| Geospatial preservation | Yes | Rasterio/pyproj | GeoTIFF | none | none | all spatial workflows | CRS/transform | Yes | Yes |
| Leakage controls | Yes | Dataset registry/evaluator | all | none | none | offline | run manifest | Yes | Yes |

The contract requires every mandatory criterion to have an implementation component, test coverage and evaluation path.


# 2. Evaluation-First Strategy

## 2.1 Priority

Unless the organisers publish different weights, engineering effort is:

```
`1. Optical-SAR joint reasoning`

`2. Change analysis`

`3. Single-image VQA`

`4. Grounding`

`5. Captioning`

`6. Agent/routing/evidence/reliability`
```

This is an **engineering priority**, not an official score ranking. The supplied specification explicitly makes optical-SAR the primary hidden-set compatibility target.


## 2.2 Evaluation dimensions

### VQA

Measure:

```
`exact match`

`normalized exact match`

`F1 where appropriate`
```

### Captioning

Use the benchmark-prescribed metrics.

Locally additionally calculate:

```
`BLEU`

`ROUGE-L`

`CIDEr`

`BERTScore`
```

### Grounding

```
`IoU`

`Recall@IoU`

`mAP where applicable`
```

### Change

```
`precision`

`recall`

`F1`

`IoU`

`mIoU`
```

### Change VQA

```
`answer accuracy`

`semantic match if benchmark specifies it`
```

### Optical-SAR

This must be **task-dependent**.

Possible evaluation:

```
`classification accuracy/F1`

`VQA accuracy`

`region IoU`

`mask IoU`

`change metrics`
```

Do not invent an official multimodal metric.


# 3. System Architecture

## 3.1 Logical architecture

```
`flowchart TD`

`    A\[User Query + Uploaded Images\]`

`    B\[Query Normalizer\]`

`    C\[Tiny NLP Intent Router\]`

`    D\[Intent Validator\]`

`    E\[Deterministic Policy Engine\]`

`    F\[Workflow DAG\]`


`    V\[VLM Specialist\]`

`    G\[Grounding Specialist\]`

`    CD\[Change Specialist\]`

`    OS\[Optical-SAR Specialist\]`

`    GEO\[Geospatial Engine\]`


`    EV\[Evidence Engine\]`

`    CF\[Confidence Calibration\]`

`    R\[Result Normalizer\]`

`    T\[Execution Trace\]`

`    REP\[Report Generator\]`


`    A --\> B`

`    B --\> C`

`    C --\> D`

`    D --\> E`

`    E --\> F`


`    F --\> V`

`    F --\> G`

`    F --\> CD`

`    F --\> OS`

`    F --\> GEO`


`    V --\> EV`

`    G --\> EV`

`    CD --\> EV`

`    OS --\> EV`

`    GEO --\> EV`


`    EV --\> CF`

`    CF --\> R`

`    R --\> T`

`    R --\> REP`

`    R --\> A`
```


## 3.2 Deployable competition architecture

```
`satquery-ai/`

`    one Python process`

`          │`

`          ├── controller`

`          ├── router`

`          ├── VLM`

`          ├── grounding`

`          ├── change`

`          ├── CROMA`

`          ├── evidence`

`          ├── evaluation`

`          └── GUI`
```

No microservices.

No Kubernetes.

No queue cluster.

No vector database.

No distributed infrastructure.

The **interfaces** are designed so those can be added later if necessary, but they are not implemented for the competition prototype.


# 4. Exact Specialist Model Stack

## 4.1 NLP intent router

### Model

`sentence-transformers/all-MiniLM-L6-v2`

License: Apache-2.0. ([Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main?utm_source=chatgpt.com))

### Architecture

```
`query`

` ↓`

`MiniLM encoder`

` ↓`

`384-d embedding`

` ↓`

`LayerNorm`

` ↓`

`Linear(384 → 128)`

` ↓`

`GELU`

` ↓`

`Dropout(0.1)`

` ↓`

`Linear(128 → task classes)`
```

Also create auxiliary heads:

```
`task`

`modality`

`temporal`

`spatial\_output`

`language\_output`
```

### Trainable portion

Only the classification adapter initially.

Keep MiniLM frozen.

This is enough because the router has a tiny, closed ontology.


# 5. Router Training Dataset

Create a **manually curated + synthetic query dataset**.

Classes:

```
`vqa`

`caption`

`grounding`

`change`

`optical\_sar`

`unsupported`
```

Generate approximately:

```
`300–500 examples/class`
```

Target:

```
`2,000–3,000 queries total`
```

Example:

```
`"Describe this image."`

`→ caption`


`"What buildings can you see?"`

`→ vqa`


`"Locate the water body."`

`→ grounding`


`"Has the urban area increased?"`

`→ change`


`"Compare optical and SAR to identify built-up regions."`

`→ optical\_sar`
```

### Hard negatives

This is important.

Include confusing examples:

```
`"Describe the changes."`
```

vs

```
`"What changed?"`
```

vs

```
`"Where did the change happen?"`
```

All map to `change`, but the latter also requires spatial output.

Similarly:

```
`"Describe the water body."`
```

→ caption/VQA

while:

```
`"Show me the water body."`
```

→ grounding.


# 6. Tiny NLP Router Decision Process

The router returns:

```
`\{`

`  "task": "grounding",`

`  "modality": "optical",`

`  "temporal": false,`

`  "spatial\_output": true,`

`  "language\_output": true,`

`  "confidence": 0.94`

`\}`
```

Then deterministic validation runs.

Example:

```
`task = grounding`

`image\_count = 1`

`modality = optical`
```

Valid.

But:

```
`task = change`

`image\_count = 1`
```

Invalid.

The policy engine changes this to:

```
`\{`

`  "status": "invalid\_request",`

`  "reason": "change analysis requires two compatible temporal inputs"`

`\}`
```

The router never overrides input reality.


# 7. Agentic Controller

The agent is therefore:

```
`NLP understanding`

`+`

`deterministic planning`

`+`

`specialist invocation`

`+`

`result aggregation`
```

It is not:

```
`LLM → arbitrary tool calling`
```

### Controller states

```
`RECEIVE`

` ↓`

`PARSE`

` ↓`

`VALIDATE`

` ↓`

`PLAN`

` ↓`

`PREPROCESS`

` ↓`

`EXECUTE`

` ↓`

`AGGREGATE`

` ↓`

`VERIFY`

` ↓`

`RESPOND`
```

Each state has allowed transitions.

This satisfies the “agentic” requirement while remaining reproducible. The project contract specifically permits and prefers constrained state-machine orchestration over unrestricted autonomous agents.


# 8. Specialist Interface

Every specialist must implement the same conceptual interface:

```
`class Specialist:`

`    name: str`

`    version: str`

`    capabilities: list\[str\]`


`    def validate\_request(self, request):`

`        ...`


`    def execute(self, request):`

`        ...`


`    def estimate\_confidence(self, output):`

`        ...`


`    def produce\_evidence(self, output):`

`        ...`
```

The implementation may be ordinary Python.

This is our **future scaling seam**.

Today:

```
`specialist.execute()`
```

Tomorrow:

```
`specialist\_worker.execute\_remote()`
```

The controller doesn't care.


# 9. Workflow Definitions

## 9.1 Workflow A — Single-image VQA

```
`Input`

` ↓`

`validate`

` ↓`

`identify modality`

` ↓`

`normalize`

` ↓`

`tile if necessary`

` ↓`

`select informative tiles`

` ↓`

`SmolVLM`

` ↓`

`answer normalization`

` ↓`

`evidence generation`

` ↓`

`confidence`

` ↓`

`result`
```

### Tile policy

Whole-image view first.

If image exceeds configured resolution:

```
`whole-image thumbnail`

`      ↓`

`candidate tiles`

`      ↓`

`top K = 4`
```

Do not send every tile through the VLM.


# 10. Workflow B — Caption

```
`Input`

` ↓`

`validation`

` ↓`

`normalization`

` ↓`

`whole-image description`

` ↓`

`optional selected tiles`

` ↓`

`deduplicate repeated objects`

` ↓`

`caption composition`

` ↓`

`confidence`

` ↓`

`result`
```


# 11. Workflow C — Grounding

This is newly promoted to a first-class workflow.

```
`Query`

` ↓`

`NLP router`

` ↓`

`grounding intent`

` ↓`

`GeoTIFF validation`

` ↓`

`image normalization`

` ↓`

`RemoteCLIP image encoder`

` ↓`

`text encoder`

` ↓`

`multiscale candidate generation`

` ↓`

`grounding head`

` ↓`

`candidate boxes`

` ↓`

`NMS`

` ↓`

`best region(s)`

` ↓`

`optional mask refinement`

` ↓`

`SmolVLM explanation`

` ↓`

`confidence`

` ↓`

`result`
```


# 12. Grounding Specialist

RemoteCLIP's official repository provides remote-sensing checkpoints for RN50, ViT-B/32 and ViT-L/14 and explicitly supports image-text semantic alignment. ([GitHub](https://github.com/ChenDelong1999/RemoteCLIP?utm_source=chatgpt.com))

### Primary checkpoint

```
`chendelong/RemoteCLIP`

`RemoteCLIP-ViT-B-32.pt`
```

Use ViT-B/32 first.

Do **not** start with ViT-L/14.

### Why?

ViT-B/32 gives:

```
`remote-sensing language alignment`

`+`

`manageable compute`

`+`

`pretrained semantics`
```

and we only need the model to generate a strong embedding/feature base.


# 13. Grounding Head

Architecture:

```
`RemoteCLIP image features`

`        │`

`        ├── global embedding`

`        │`

`        └── patch features`

`               │`

`        multi-scale projector`

`               │`

`       text/image similarity`

`               │`

`       region scoring head`

`               │`

`          box regression`

`               │`

`             NMS`
```

### Initial head

```
`Linear(D → 512)`

`GELU`

`LayerNorm`

`Linear(512 → 256)`

`GELU`

`Linear(256 → 5)`
```

Output:

```
`x1`

`y1`

`x2`

`y2`

`confidence`
```

Use normalized coordinates internally:

```
`0–1`
```

and convert to:

```
`pixel coordinates`
```

for output.

VRSBench's current repository notes that provided evaluation box coordinates are normalized to 0–100, so the evaluator adapter must explicitly convert between its coordinate convention and our internal 0–1 representation rather than quietly treating the numbers as pixels. ([GitHub](https://github.com/lx709/VRSBench?utm_source=chatgpt.com))


# 14. Grounding Training Strategy

Use VRSBench referring expressions.

Current VRSBench documentation reports:

- 29,614 images

- 52,472 object references

- 123,221 VQA pairs

- human-verified captions and referring annotations. ([GitHub](https://github.com/lx709/VRSBench?utm_source=chatgpt.com))

### Training split

Never touch evaluation data while training.

Create:

```
`train`

`validation`

`immutable test`
```

at image/scene level.

### Training

Freeze RemoteCLIP.

Train:

```
`projection`

`grounding head`
```

Loss:

```
`L = 0.5 \* L1\_box`

`  + 0.3 \* GIoU`

`  + 0.2 \* BCE\_confidence`
```

Initial learning rate:

```
`1e-4`
```

Batch:

```
`16–32`
```

CPU preprocessing + GPU model.

This should be cheap compared with VLM fine-tuning.


# 15. Optional Grounding Mask Refinement

Do not make segmentation a dependency.

If a lightweight segmentation model becomes available and fits the budget:

```
`box`

` ↓`

`mask refinement`
```

Otherwise:

```
`box`

` ↓`

`polygon rectangle`
```

The mandatory grounding contract is satisfied by accurate spatial localization.


# 16. Optical-SAR Architecture

CROMA remains the primary model.

The official implementation uses:

```
`SAR encoder`

`optical encoder`

`joint cross-modal representation`
```

and the base model uses:

```
`768 dimensional feature representation`

`12-channel optical input`

`2-channel SAR input`
```

with a 120×120 default pretraining image size. ([GitHub](https://github.com/antofuller/CROMA/blob/main/README.md?utm_source=chatgpt.com))


# 17. Optical-SAR Fusion

Use:

```
`CROMA optical feature`

`CROMA SAR feature`

`CROMA joint feature`
```

then:

```
`concat`

` ↓`

`LayerNorm`

` ↓`

`Linear`

` ↓`

`GELU`

` ↓`

`Dropout`

` ↓`

`task head`
```

### Do not use

```
`RGB image + colorized SAR`
```

as the final fusion architecture.

That is visually convincing and scientifically rather shallow.


# 18. CROMA Sensor Adapter

CROMA's pretrained architecture is explicitly Sentinel-1/Sentinel-2 oriented. ([GitHub](https://github.com/antofuller/CROMA/blob/main/README.md?utm_source=chatgpt.com))

The hidden evaluation target described in your specification is Cartosat-2S + RISAT.

Therefore create:

```
`sensor\_adapter.py`
```

with:

```
`sensor`

`band\_map`

`normalization`

`availability\_mask`

`resolution`
```

Example:

```
`\{`

`  "sensor": "cartosat\_2s",`

`  "available\_bands": \["B1", "B2", "B3", "B4"\],`

`  "mapping": \{`

`    "B1": "c1",`

`    "B2": "c2",`

`    "B3": "c3",`

`    "B4": "c4"`

`  \}`

`\}`
```

No invented missing bands.

Missing channels become masked/zero-filled according to the validated adapter policy.


# 19. RISAT Handling

RISAT imagery may vary in acquisition/polarization characteristics.

ISRO documentation describes RISAT-1 as a C-band SAR mission with multiple polarization configurations. Therefore the SAR adapter must inspect actual available channels rather than assuming one fixed polarization pair.

### Internal representation

```
`canonical\_sar\[2, H, W\]`

`+`

`sar\_channel\_mask\[2\]`
```

Then CROMA receives both:

```
`SAR tensor`

`channel availability metadata`
```


# 20. Sensor Robustness Training

During fusion-head training randomly mask inputs:

```
`optical:`

`100%`

`80%`

`60%`

`40%`


`SAR:`

`100%`

`50%`
```

This creates robustness to modality differences and missing channels.

Do not synthesize fake SAR data.

Do not fabricate spectral bands.


# 21. Multitemporal Change Architecture

```
`T1`

`T2`

` ↓`

`validation`

` ↓`

`CRS comparison`

` ↓`

`resolution comparison`

` ↓`

`co-registration check`

` ↓`

`shared tiling`

` ↓`

`shared encoder`

` ↓`

`feature difference`

` ↓`

`attention/change representation`

` ↓`

`decoder`

` ↓`

`probability map`

` ↓`

`morphological cleanup`

` ↓`

`connected components`

` ↓`

`change regions`

` ↓`

`SmolVLM interpretation`
```

STANet's official implementation provides a practical Siamese spatial-temporal attention approach for bitemporal remote-sensing change detection and the LEVIR-CD dataset workflow. ([GitHub](https://github.com/ChenDelong1999/RemoteCLIP?utm_source=chatgpt.com))


# 22. Change Model

Use a compact ResNet18-class Siamese encoder.

Architecture:

```
`T1 ──► Encoder ──► F1`

`                │`

`T2 ──► Encoder ──► F2`


`F1 + F2`

`   ↓`

`difference/attention`

`   ↓`

`decoder`

`   ↓`

`change probability`
```

Shared encoder weights.

### Loss

```
`0.5 BCE`

`+`

`0.5 Dice`
```

Tune around:

```
`0.25–0.75 BCE weighting`
```

on validation.


# 23. False Change Handling

### Illumination

Use learned features rather than raw pixel subtraction.

### Registration

Calculate registration quality.

### Season

Augment brightness/contrast/radiometric scaling during training.

### Cloud/nodata

Maintain an invalid-data mask.

### Sensor differences

Normalize each acquisition separately.

### Low-quality alignment

Reduce confidence and optionally refuse spatial conclusions.

Never convert poor registration into fake certainty.


# 24. Evidence System

The evidence engine is shared across all specialists.

## Evidence types

```
`IMAGE\_CROP`

`TILE`

`BOUNDING\_BOX`

`MASK`

`CHANGE\_MAP`

`OPTICAL\_VIEW`

`SAR\_VIEW`

`JOINT\_FEATURE\_REGION`

`STATISTIC`

`GEOLOCATION`
```

Each evidence item has:

```
`\{`

`  "id": "evidence\_003",`

`  "type": "bounding\_box",`

`  "source": "grounding",`

`  "coordinates": \[0.21, 0.33, 0.47, 0.61\],`

`  "score": 0.87`

`\}`
```


# 25. VLM and Evidence Relationship

Do not ask SmolVLM to hallucinate coordinates.

Instead:

```
`specialist`

` ↓`

`real evidence`

` ↓`

`VLM`

` ↓`

`language explanation`
```

The VLM gets evidence references.

Example:

```
`Grounding model:`

`water\_region = \[x1,y1,x2,y2\]`

`confidence=0.91`
```

Then VLM says:

> The highlighted region corresponds to the water body.

The region came from a specialist.

The sentence came from the language model.

That distinction matters.


# 26. Confidence System

No LLM-generated confidence.

### Router confidence

Softmax probability.

### Grounding

```
`box confidence`

`+`

`text/image similarity`

`+`

`augmentation consistency`
```

### Change

```
`mean pixel probability`

`+`

`component stability`

`+`

`registration quality`
```

### Optical-SAR

```
`fusion classifier margin`

`+`

`optical confidence`

`+`

`SAR confidence`

`+`

`cross-modal agreement`
```

Then calibration:

```
`raw score`

` ↓`

`validation-calibrated mapping`

` ↓`

`final confidence`
```

Use temperature scaling where appropriate.


# 27. Execution Trace

```
`\{`

`  "run\_id": "uuid",`

`  "query": "...",`

`  "router": \{`

`    "task": "grounding",`

`    "confidence": 0.94`

`  \},`

`  "validation": \{`

`    "input\_count": 1,`

`    "format": "passed",`

`    "modality": "optical"`

`  \},`

`  "workflow": \[`

`    "validate",`

`    "preprocess",`

`    "ground",`

`    "evidence",`

`    "confidence",`

`    "answer"`

`  \],`

`  "models": \[`

`    \{`

`      "name": "RemoteCLIP",`

`      "revision": "..."`

`    \},`

`    \{`

`      "name": "SmolVLM",`

`      "revision": "..."`

`    \}`

`  \],`

`  "outputs": \[`

`    "bbox",`

`    "answer"`

`  \],`

`  "confidence": 0.87,`

`  "timings": \{\},`

`  "fallbacks": \[\]`

`\}`
```

No chain-of-thought.

Only observable execution facts.


# 28. Natural-Language Router + Deterministic Orchestration

This is the answer to your original idea.

### The router is allowed to answer:

```
`What does the user want?`
```

### The controller answers:

```
`What inputs are valid?`

`Which workflow is legal?`

`Which specialists run?`

`In what order?`

`What parameters are permitted?`

`How are outputs combined?`
```

That means the system genuinely understands arbitrary phrasing without surrendering control.


# 29. Example Query Routing

### Query

> “Can you show me where the water body is?”

Router:

```
`\{`

`  "task": "grounding",`

`  "spatial\_output": true`

`\}`
```

Workflow:

```
`Grounding`

`→ box`

`→ evidence`

`→ VLM explanation`
```

### Query

> “Describe this satellite image.”

```
`caption`
```

### Query

> “What is the dominant land cover?”

```
`vqa`
```

### Query

> “What changed?”

```
`change`
```

### Query

> “Compare the radar and optical images to locate built-up areas.”

```
`optical\_sar`
```

### Query

> “What changed and show me where?”

```
`change`

`+`

`spatial\_output=true`
```

One query can therefore result in a multi-step DAG.


# 30. Dynamic Workflow DAG

The router outputs **intent attributes**, not a fixed tool call.

Example:

```
`\{`

`  "task": "change",`

`  "spatial\_output": true,`

`  "language\_output": true,`

`  "evidence\_required": true`

`\}`
```

The policy engine constructs:

```
`validate`

` ↓`

`align`

` ↓`

`change detector`

` ↓`

`change regions`

` ↓`

`VLM explanation`

` ↓`

`confidence`

` ↓`

`evidence`
```

This is more powerful than static routing while remaining deterministic.


# 31. Dataset Engineering

## BigEarthNet v2

Use for:

- remote-sensing adaptation

- land-cover understanding

- optical/SAR paired learning

- robustness experiments

The official BigEarthNet site reports 549,488 Sentinel-1/Sentinel-2 patch pairs in v2.0.

### Subset

Start:

```
`50,000 training`

`5,000 validation`
```

Then scale if the Kaggle budget permits.


# 32. VRSBench

Use for:

```
`captioning`

`VQA`

`grounding`
```

Current repository information reports:

```
`29,614 images`

`29,614 human-verified captions`

`52,472 object references`

`123,221 VQA pairs`
```

and provides evaluation code. ([GitHub](https://github.com/lx709/VRSBench?utm_source=chatgpt.com))


# 33. RSVQA

Use for:

```
`single-image VQA`
```

Keep benchmark-specific preprocessing separate from generic EO preprocessing.

Never mix benchmark-specific conventions into the common image loader.


# 34. CDVQA

Use for:

```
`change-based VQA`

`change description`

`temporal reasoning`
```

Keep its annotation schema in an adapter.


# 35. LEVIR-CD

Use for:

```
`binary change-mask training`
```

Training patch size:

```
`256×256`
```

in the standard STANet workflow.


# 36. Data Leakage Prevention

This remains one of the hardest requirements.

## Every sample gets:

```
`dataset\_id`

`scene\_id`

`sample\_id`

`source\_scene`

`geographic\_hash`

`acquisition\_date`

`sensor`

`sha256`

`split`
```

### Split by scene

Never randomly split neighboring patches.

```
`scene`

` ↓`

`train OR validation OR test`
```

not:

```
`patch`

` ↓`

`random split`
```


# 37. Test Set Firewall

Public test sets:

```
`evaluation/public\_test/`
```

are immutable.

Training code may not import them.

Evaluation code may read them only in:

```
`evaluation mode`
```

No cache of test answers is permitted.


# 38. Hidden Set Firewall

There must be no:

```
`hidden\_test\_mode`
```

in training.

The hidden evaluation interface accepts:

```
`input imagery`
```

and produces:

```
`standard result schema`
```

with no knowledge of the hidden answer.


# 39. GeoTIFF Contract

Validation:

```
`file`

` ↓`

`TIFF/GeoTIFF parser`

` ↓`

`dimensions`

` ↓`

`bands`

` ↓`

`dtype`

` ↓`

`CRS`

` ↓`

`transform`

` ↓`

`bounds`

` ↓`

`nodata`

` ↓`

`modality`

` ↓`

`temporal metadata`

` ↓`

`pair compatibility`
```


# 40. Pair Compatibility

For optical-SAR:

```
`CRS compatible?`

`bounds overlap?`

`dimensions sensible?`

`resolution known?`

`co-registration metadata?`
```

For temporal:

```
`T1 != T2`

`same geographic scene?`

`overlap?`

`resolution compatibility?`

`alignment quality?`
```


# 41. Preprocessing Configuration

```
`image:`

`  max\_pixels: 25000000`

`  tile\_size: 512`

`  tile\_overlap: 128`

`  max\_tiles: 64`


`optical:`

`  normalization: percentile`

`  lower\_percentile: 2`

`  upper\_percentile: 98`


`sar:`

`  representation: db`

`  clip\_min\_db: -30`

`  clip\_max\_db: 5`


`change:`

`  tile\_size: 256`

`  tile\_overlap: 32`


`grounding:`

`  max\_candidates: 20`

`  nms\_iou: 0.5`
```


# 42. Large Image Handling

```
`large image`

` ↓`

`overview`

` ↓`

`candidate tile scoring`

` ↓`

`top 4 tiles`

` ↓`

`specialist inference`
```

Candidate scoring can use cheap visual statistics:

```
`edge density`

`entropy`

`CLIP similarity`

`objectness`
```

No VLM inference for every tile.


# 43. Remote-Sensing Adaptation

The official project requires adaptation using BigEarthNet or other allowed remote-sensing data.

### VLM adaptation

Freeze:

```
`vision backbone`

`most language weights`
```

Train:

```
`LoRA adapters`
```

### Initial configuration

```
`lora:`

`  rank: 16`

`  alpha: 32`

`  dropout: 0.05`


`training:`

`  lr: 2e-4`

`  effective\_batch\_size: 16`

`  epochs: 1`

`  warmup\_ratio: 0.05`

`  weight\_decay: 0.01`

`  precision: bf16`
```


# 44. VLM Training Data Construction

From BigEarthNet labels create controlled instruction pairs.

Example:

```
`Image:`

`BigEarthNet patch`


`Question:`

`"What land-cover categories are visible?"`


`Answer:`

`"Arable land, mixed forest and urban fabric."`
```

Also:

```
`"Is urban land cover visible?"`

`"Is water present?"`

`"Which category dominates?"`

`"Describe the scene using land-cover labels."`
```

Do not manufacture precise object counts that the source annotations do not support.


# 45. Kaggle Budget

Current Kaggle documentation says free notebook GPU use includes T4×2, and the P100 has now been retired effective September 15, 2026. Kaggle currently documents 12-hour CPU/GPU notebook sessions, and its efficient-GPU documentation describes a weekly GPU quota around 30 hours, subject to availability. ([Kaggle](https://www.kaggle.com/docs/notebooks?utm_source=chatgpt.com))

Therefore the plan must now assume:

**T4×2 first.**

Not P100.

That is a material update from the earlier plan.


# 46. Kaggle Training Schedule

| **Stage** | **Accelerator** | **Training** | **Target** |
| :-: | :-: | :-: | :-: |
| Router | CPU/T4 | ~2–3k queries | \<1 h |
| VLM adaptation | T4×2 | 40–60k samples | ≤8 h |
| Grounding head | T4×2 | VRSBench refs | ≤4 h |
| Change detector | T4×2 | LEVIR-CD | ≤5 h |
| Change VQA | T4×2 | CDVQA subset | ≤3 h |
| CROMA fusion head | T4×2 | BigEarthNet pair subset | ≤3 h |
| Calibration | CPU | validation | \<1 h |

These are **budget targets**, not measured benchmark timings.

Every actual training run records:

```
`wall\_time`

`peak\_VRAM`

`peak\_RAM`

`samples\_per\_second`

`checkpoint\_size`
```


# 47. Training Contingency

If a stage exceeds budget:

```
`1. reduce dataset subset`

`2. freeze additional weights`

`3. lower resolution`

`4. reduce batch`

`5. increase gradient accumulation`

`6. reduce LoRA rank`

`7. reduce epochs`
```

Never increase architectural complexity as the first response to a compute problem.


# 48. Hugging Face Deployment

Hugging Face currently documents ZeroGPU Spaces as Gradio-only, with free personal accounts able to host up to two ZeroGPU Spaces, and a current free quota of 5 GPU minutes/day. The default `large` configuration provides 48 GB VRAM. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu?utm_source=chatgpt.com))

For the competition prototype:

```
`One Gradio Space`
```

with:

```
`CPU preprocessing`

`+`

`GPU specialist execution when available`
```

### Important

ZeroGPU is an accelerator.

It is not our architectural dependency.


# 49. Lazy Model Loading

Use a simple model manager:

```
`request`

` ↓`

`determine specialists`

` ↓`

`load only those specialists`

` ↓`

`execute`

` ↓`

`release unused models`
```

Do not permanently place every model on GPU.


# 50. Deployment Model Footprint Strategy

Keep:

```
`SmolVLM`

`RemoteCLIP ViT-B/32`

`STANet`

`CROMA-base`

`MiniLM router`
```

but load them according to workflow.

### VQA

```
`MiniLM`

`SmolVLM`
```

### Grounding

```
`MiniLM`

`RemoteCLIP`

`SmolVLM`
```

### Change

```
`MiniLM`

`STANet`

`SmolVLM`
```

### Optical-SAR

```
`MiniLM`

`CROMA`

`fusion head`

`SmolVLM`
```

This is why the modular specialist architecture matters.


# 51. GUI

## Header

```
`SatQuery AI`

`Interactive Remote-Sensing Intelligence`
```

### Upload area

```
`Single Image`

`T1 / T2`

`Optical`

`SAR`
```

### Viewer

Tabs:

```
`Original`

`Evidence`

`Grounding`

`Change`

`Optical`

`SAR`

`Fusion`
```

### Query

```
`\[ Ask SatQuery AI... \]`


`Detected task:`

`Optical + SAR`


`Running:`

`✓ validation`

`✓ optical encoder`

`✓ SAR encoder`

`→ fusion`
```


# 52. Result View

```
`Answer`

`--------------------------------`

`Built-up regions are concentrated`

`in the eastern portion of the image.`


`Confidence`

`72%`


`Evidence`

`\[highlighted image\]`


`Optical evidence`

`\[SAR evidence\]`


`Execution`

`\[details\]`


`Download`

`\[JSON\] \[PDF\]`
```

The evaluator should be able to discover the entire architecture by using the application.


# 53. Result Schema

Final schema:

```
`\{`

`  "task": "vqa|caption|grounding|change|optical\_sar",`

`  "answer": "...",`

`  "labels": \[\],`

`  "regions": \[\],`

`  "boxes": \[\],`

`  "masks": \[\],`

`  "change\_map": null,`

`  "evidence": \[\],`

`  "confidence": \{\},`

`  "geospatial": \{\},`

`  "execution\_trace": \{\}`

`\}`
```

The supplied project schema already establishes the core fields. Grounding adds `regions` to make spatial outputs first-class.


# 54. API Contracts

## Query

```
`\{`

`  "assets": \["asset\_001"\],`

`  "query": "Show the water body."`

`\}`
```

## Intent

```
`\{`

`  "task": "grounding",`

`  "modality": "optical",`

`  "spatial\_output": true,`

`  "confidence": 0.92`

`\}`
```

## Analysis

```
`\{`

`  "workflow": "grounding",`

`  "assets": \["asset\_001"\],`

`  "config\_hash": "..."`

`\}`
```

## Result

Use the master result schema.

## Trace

Use the execution trace schema.

## Health

```
`\{`

`  "status": "ok",`

`  "models": \{`

`    "vlm": "ready",`

`    "grounding": "ready",`

`    "change": "ready",`

`    "fusion": "ready"`

`  \}`

`\}`
```


# 55. Repository Structure

```
`satquery-ai/`

`│`

`├── app/`

`│   ├── gradio\_app.py`

`│   ├── ui\_state.py`

`│   └── visualizers.py`

`│`

`├── core/`

`│   ├── config.py`

`│   ├── schemas.py`

`│   ├── controller.py`

`│   ├── planner.py`

`│   ├── registry.py`

`│   └── errors.py`

`│`

`├── router/`

`│   ├── encoder.py`

`│   ├── classifier.py`

`│   ├── adapter.py`

`│   ├── train.py`

`│   └── dataset.py`

`│`

`├── specialists/`

`│   ├── base.py`

`│   │`

`│   ├── vqa/`

`│   │   ├── model.py`

`│   │   ├── inference.py`

`│   │   └── prompts.py`

`│   │`

`│   ├── grounding/`

`│   │   ├── remoteclip.py`

`│   │   ├── head.py`

`│   │   ├── inference.py`

`│   │   └── postprocess.py`

`│   │`

`│   ├── change/`

`│   │   ├── stanet.py`

`│   │   ├── inference.py`

`│   │   └── postprocess.py`

`│   │`

`│   └── optical\_sar/`

`│       ├── croma.py`

`│       ├── fusion\_head.py`

`│       └── inference.py`

`│`

`├── preprocessing/`

`│   ├── raster.py`

`│   ├── optical.py`

`│   ├── sar.py`

`│   ├── tiling.py`

`│   ├── temporal.py`

`│   └── sensor\_adapter.py`

`│`

`├── geospatial/`

`│   ├── crs.py`

`│   ├── alignment.py`

`│   ├── transform.py`

`│   └── overlap.py`

`│`

`├── evidence/`

`│   ├── boxes.py`

`│   ├── masks.py`

`│   ├── crops.py`

`│   ├── change\_maps.py`

`│   └── confidence.py`

`│`

`├── reports/`

`│   └── generator.py`

`│`

`├── evaluation/`

`│   ├── manifests.py`

`│   ├── leakage.py`

`│   ├── metrics/`

`│   ├── normalize.py`

`│   ├── runner.py`

`│   └── benchmark\_adapters/`

`│`

`├── training/`

`│   ├── vlm/`

`│   ├── grounding/`

`│   ├── change/`

`│   ├── fusion/`

`│   └── calibration/`

`│`

`├── configs/`

`│   ├── base.yaml`

`│   ├── train.yaml`

`│   ├── eval.yaml`

`│   └── deploy.yaml`

`│`

`├── tests/`

`│   ├── unit/`

`│   ├── routing/`

`│   ├── geospatial/`

`│   ├── leakage/`

`│   ├── model/`

`│   └── e2e/`

`│`

`├── scripts/`

`│   ├── prepare\_data.py`

`│   ├── create\_manifest.py`

`│   ├── train\_router.py`

`│   ├── train\_vlm.py`

`│   ├── train\_grounding.py`

`│   ├── train\_change.py`

`│   ├── train\_fusion.py`

`│   └── evaluate.py`

`│`

`└── README.md`
```


# 56. Central Configuration Registry

```
`project:`

`  name: satquery-ai`

`  version: "1.0.0"`

`  seed: 42`


`image:`

`  max\_pixels: 25000000`

`  tile\_size: 512`

`  tile\_overlap: 128`

`  max\_tiles: 64`


`router:`

`  model: sentence-transformers/all-MiniLM-L6-v2`

`  max\_length: 128`

`  hidden\_dim: 128`

`  dropout: 0.10`

`  confidence\_threshold: 0.70`


`vlm:`

`  checkpoint: HuggingFaceTB/SmolVLM-500M-Instruct`

`  max\_new\_tokens: 128`

`  temperature: 0.0`


`grounding:`

`  checkpoint: chendelong/RemoteCLIP`

`  variant: ViT-B-32`

`  nms\_iou: 0.50`

`  max\_candidates: 20`

`  confidence\_threshold: 0.40`


`change:`

`  tile\_size: 256`

`  tile\_overlap: 32`

`  threshold: 0.50`

`  min\_component\_pixels: 32`


`croma:`

`  checkpoint: antofuller/CROMA`

`  variant: base`

`  image\_resolution: 120`

`  optical\_channels: 12`

`  sar\_channels: 2`


`optical:`

`  normalization: percentile`

`  lower\_percentile: 2`

`  upper\_percentile: 98`


`sar:`

`  representation: db`

`  clip\_min\_db: -30`

`  clip\_max\_db: 5`


`training:`

`  precision: bf16`

`  vlm\_lr: 0.0002`

`  vlm\_batch\_size: 2`

`  vlm\_gradient\_accumulation: 8`

`  lora\_rank: 16`

`  lora\_alpha: 32`

`  lora\_dropout: 0.05`

`  weight\_decay: 0.01`

`  warmup\_ratio: 0.05`

`  epochs: 1`


`grounding\_training:`

`  learning\_rate: 0.0001`

`  box\_loss\_weight: 0.5`

`  giou\_loss\_weight: 0.3`

`  confidence\_loss\_weight: 0.2`


`confidence:`

`  temperature\_scaling: true`


`runtime:`

`  max\_specialists: 4`

`  timeout\_seconds: 120`

`  unload\_after\_workflow: true`


`evaluation:`

`  immutable\_public\_test: true`

`  hidden\_data\_access: false`

`  official\_aggregate\_weights: null`
```


# 57. Failure Matrix

| **Failure** | **Detection** | **Recovery** | **User message** | **Trace** | **Fallback** |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Router low confidence | probability threshold | ask classification fallback rules | “I’m not confident what task you requested.” | yes | deterministic keyword layer |
| Router predicts invalid task | schema validator | reject | “The uploaded inputs do not support this task.” | yes | none |
| Corrupt TIFF | rasterio | reject | unreadable | yes | none |
| Missing CRS | metadata parser | degraded | CRS missing | yes | non-geospatial mode |
| Pair misalignment | alignment test | reject spatial analysis | images not sufficiently aligned | yes | textual only if safe |
| Grounding head fails | runtime | retry CPU/GPU | grounding unavailable | yes | VQA answer without box |
| Change model fails | runtime | fallback raw feature difference | change detector unavailable | yes | reduced mode |
| CROMA unavailable | runtime | modality-specific analysis | joint fusion unavailable | yes | optical/SAR separate |
| VLM OOM | exception | lower resolution | reduced reasoning mode | yes | template result |
| Timeout | timer | abort specialist | processing timeout | yes | partial result |
| Low confidence | calibration | cautious answer | low confidence | yes | uncertainty text |


# 58. Fallback Router

Do not make the system entirely dependent on the trained intent model.

Use:

```
`Tiny NLP router`

`        ↓`

`if confidence ≥ threshold`

`    accept`

`else`

`    deterministic lexical fallback`
```

Examples:

```
`"where"`

`"locate"`

`"highlight"`

`"show region"`
```

→ grounding

```
`"changed"`

`"between"`

`"before and after"`

`"temporal"`
```

→ change

```
`"optical and SAR"`

`"radar and optical"`

`"both images"`
```

→ optical-SAR

This does **not** replace the learned router.

It protects against obvious failures.


# 59. Testing

## Router tests

At least:

```
`500 validation queries`

`100 hard negatives`

`50 unsupported queries`
```

Target:

```
`task accuracy ≥ 95%`
```

This is an engineering acceptance threshold, not a benchmark claim.


## Grounding tests

```
`IoU`

`Recall@0.5`

`coordinate conversion`

`NMS`
```

Test:

```
`object in center`

`object at boundary`

`multiple objects`

`small objects`

`large objects`
```


## Change tests

```
`perfect alignment`

`1-pixel shift`

`10-pixel shift`

`brightness change`

`no change`

`large construction change`

`small change`
```


# 60. Adversarial Tests

The application must survive:

```
`blank image`

`all-zero image`

`extremely bright image`

`extremely dark image`

`noise image`

`unsupported TIFF`

`10000×10000 image`

`missing metadata`

`wrong modality labels`

`same image uploaded twice`

`one temporal image`

`three images`

`optical/SAR size mismatch`
```


# 61. Evaluation Harness

Architecture:

```
`Dataset Adapter`

` ↓`

`Immutable Manifest`

` ↓`

`Prediction Runner`

` ↓`

`Schema Validator`

` ↓`

`Metrics`

` ↓`

`Normalization`

` ↓`

`Report`
```


# 62. Public/Hidden Evaluation Modes

### Public mode

```
`benchmark test`

`frozen model`

`frozen config`

`frozen prompts`
```

### Hidden-compatible mode

```
`unknown imagery`

`same preprocessing`

`same controller`

`same result schema`

`same model pipeline`
```

No benchmark-specific branching.


# 63. Metric Normalisation

For each raw metric:

```
`raw`

` ↓`

`task-specific transform`

` ↓`

`0–1 normalized`
```

The framework supports:

```
`aggregate:`

`  official\_weights: null`
```

until the organizers publish them.

The uploaded specification explicitly prohibits inventing an official aggregate formula.


# 64. Ablation Studies

Minimum set:

### VLM

```
`base SmolVLM`

`vs`

`RS-adapted SmolVLM`
```

### Fusion

```
`optical only`

`SAR only`

`CROMA joint`
```

### Change

```
`raw difference`

`vs`

`STANet-style model`
```

### Grounding

```
`RemoteCLIP zero-shot`

`vs`

`grounding head`
```

### Router

```
`keyword baseline`

`vs`

`MiniLM router`
```

### Agent

```
`hardcoded workflows`

`vs`

`NLP + deterministic policy`
```

This last ablation is particularly useful because it demonstrates that the “agentic” component is actually useful without pretending it is magic.


# 65. Research/Architecture Search Plan

| **Candidate** | **Purpose** | **Decision** |
| :-: | :-: | :-: |
| SmolVLM-500M | compact VQA/caption | **Primary** |
| PaliGemma2-3B | stronger VLM research baseline | Benchmark fallback |
| RemoteCLIP ViT-B/32 | grounding | **Primary** |
| RemoteCLIP ViT-L/14 | stronger grounding | Optional |
| CROMA-base | optical-SAR | **Primary** |
| Prithvi-EO-2.0-300M | optical EO features | Alternative |
| STANet-style | change | **Primary** |
| SatMAE | temporal representation | Reject for primary path |
| SmolLM2-135M | NLP router | Not primary |
| MiniLM-L6-v2 | NLP router | **Primary** |

PaliGemma 2 has out-of-the-box capabilities spanning VQA, captioning, object detection and segmentation, making it a useful research comparison, but its 3B-scale footprint and licensing/access conditions make it less attractive for the final compact competition runtime. ([Hugging Face](https://huggingface.co/google/paligemma2-3b-mix-224?utm_source=chatgpt.com))

SmolLM2-135M is Apache-2.0 and tiny, but it is a causal language model, so using it solely to classify a six-class intent space is needless generative machinery. ([Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json?utm_source=chatgpt.com))


# 66. Decision Log

| **Decision** | **Chosen option** | **Alternatives** | **Reason** |
| :-: | :-: | :-: | :-: |
| Router | MiniLM + classifier adapter | SmolLM2-135M | classification is simpler |
| Orchestration | NLP + deterministic policy | autonomous agent | reproducibility |
| VLM | SmolVLM-500M | PaliGemma2 | compact runtime |
| Grounding | RemoteCLIP B/32 + head | VLM coordinate generation | actual spatial model |
| Secondary task | captioning | grounding only | now grounding additionally exists |
| Change | STANet-style | raw difference | learned change representation |
| Fusion | CROMA | CNN concatenation | native optical-SAR representation |
| Training | frozen backbones + heads/LoRA | full FT | compute |
| Deployment | Gradio modular monolith | microservices | competition scope |


# 67. Immutable Decisions

The implementation model must not redesign:

```
`\[ \] modular-monolith architecture`

`\[ \] tiny NLP intent router`

`\[ \] deterministic policy engine`

`\[ \] common specialist interface`

`\[ \] SmolVLM VLM layer`

`\[ \] RemoteCLIP grounding path`

`\[ \] STANet-style change path`

`\[ \] CROMA optical-SAR path`

`\[ \] shared evidence engine`

`\[ \] calibrated confidence`

`\[ \] common result schema`

`\[ \] execution trace`

`\[ \] leakage isolation`
```


# 68. Variables Open for Tuning

```
`LoRA rank`

`VLM learning rate`

`router adapter dimension`

`router confidence threshold`

`grounding head architecture`

`grounding learning rate`

`change threshold`

`change loss weighting`

`CROMA fusion head width`

`tile size`

`tile overlap`

`top-K tile count`

`confidence calibration temperature`
```


# 69. Variables Requiring Experimental Optimization

## Router

```
`hidden dimension:`

`64–256`


`dropout:`

`0–0.3`


`confidence:`

`0.60–0.90`
```

## Grounding

```
`head width:`

`256–1024`


`learning rate:`

`5e-5–2e-4`


`NMS:`

`0.4–0.6`
```

## Change

```
`threshold:`

`0.30–0.70`


`minimum component:`

`16–128 px`
```

## CROMA

```
`fusion width:`

`256–1024`


`dropout:`

`0–0.3`
```

Selection:

**validation only.**


# 70. Risk Register

| **Risk** | Probability | Impact | **Mitigation** |
| :-: | -: | -: | :-: |
| VLM insufficient EO reasoning | Medium | High | RS adaptation + stronger benchmark |
| Grounding weak on overhead imagery | Medium | High | VRSBench adaptation + RemoteCLIP |
| CROMA Sentinel-to-Indian-sensor shift | High | Very High | sensor adapter + channel dropout |
| Hidden test distribution shift | High | Very High | sensor-agnostic preprocessing |
| Kaggle quota | Medium | High | staged training/checkpoints |
| GPU OOM | Medium | High | frozen models + PEFT |
| leakage | Low/Medium | Catastrophic | scene-level manifests |
| router mistakes | Low | Medium | confidence + deterministic fallback |
| model unavailable | Low | Medium | local HF cache + pinned revisions |
| deployment memory | Medium | High | lazy loading |


# 71. Implementation Order

This is the actual build order I would give the coding model.

## Phase 0

Environment and repository skeleton.

## Phase 1

Config registry + typed schemas.

## Phase 2

GeoTIFF loading and validation.

## Phase 3

Dataset manifests + leakage framework.

## Phase 4

MiniLM intent router.

## Phase 5

SmolVLM baseline.

## Phase 6

BigEarthNet VLM adaptation.

## Phase 7

RemoteCLIP grounding baseline.

## Phase 8

Grounding head training.

## Phase 9

STANet change detector.

## Phase 10

CDVQA reasoning integration.

## Phase 11

CROMA integration.

## Phase 12

Optical-SAR fusion head.

## Phase 13

Evidence engine.

## Phase 14

Confidence calibration.

## Phase 15

Deterministic workflow controller.

## Phase 16

GUI.

## Phase 17

Benchmark evaluation harness.

## Phase 18

HF deployment.

## Phase 19

Final hardening and demonstration.


# 72. Stop/Go Gates

## Gate 1

Proceed only when:

```
`\[ \] TIFF validator works`

`\[ \] CRS handling works`

`\[ \] leakage test passes`
```

## Gate 2

Proceed to VLM adaptation only when:

```
`\[ \] baseline inference works`

`\[ \] dataset manifests frozen`
```

## Gate 3

Proceed to grounding training when:

```
`\[ \] RemoteCLIP loads`

`\[ \] VRSBench boxes parse correctly`

`\[ \] coordinate conversions tested`
```

## Gate 4

Proceed to controller integration when:

```
`\[ \] VQA works`

`\[ \] grounding works`

`\[ \] change works`

`\[ \] CROMA works`
```

## Gate 5

Proceed to final evaluation when:

```
`\[ \] all prompts frozen`

`\[ \] all thresholds frozen`

`\[ \] model revisions frozen`

`\[ \] config hash recorded`

`\[ \] public test isolation verified`
```


# 73. "DO NOT BUILD THIS"

Do not build:

```
`\[ \] Kubernetes`

`\[ \] Docker swarm`

`\[ \] Kafka`

`\[ \] Redis cluster`

`\[ \] vector database`

`\[ \] autonomous agent swarm`

`\[ \] multi-LLM debate`

`\[ \] seven VLMs`

`\[ \] full CROMA retraining`

`\[ \] full foundation-model retraining`

`\[ \] LLM-generated bounding boxes`

`\[ \] LLM-generated confidence`

`\[ \] benchmark-specific hidden-test branches`

`\[ \] cloud inference dependency`
```

The architecture is intentionally **competition-scale**.


# 74. Production Readiness Boundary

You explicitly said **do not add production implementations**.

So this plan deliberately stops here:

### Included

```
`modularity`

`stable interfaces`

`lazy model loading`

`error handling`

`config centralization`

`traceability`

`reproducibility`

`checkpointing`

`evaluation`

`basic resource limits`
```

### Not included

```
`authentication`

`multi-tenant isolation`

`distributed queues`

`autoscaling`

`observability platform`

`Kubernetes`

`service mesh`

`distributed storage`

`horizontal worker orchestration`

`enterprise security`

`billing`

`SLA infrastructure`
```

The architecture is **scale-compatible**, but not a production implementation.

That's exactly what we want at this stage.


# 75. Final Feasibility Audit

### Can it train on current Kaggle free GPU?

**YES, conditionally.**

Current Kaggle documentation now points to T4×2 as the available newer GPU path, with the P100 retired on September 15, 2026. ([Kaggle](https://www.kaggle.com/product-announcements/735239?utm_source=chatgpt.com))

### Can it deploy on HF free hosting?

**YES, conditionally.**

HF currently supports free ZeroGPU Spaces for eligible personal accounts, with up to two Spaces and a free daily GPU quota. ([Hugging Face](https://huggingface.co/docs/hub/spaces-zerogpu?utm_source=chatgpt.com))

### Remote-sensing adaptation?

**YES.**

### Single-image VQA?

**YES.**

### Captioning?

**YES.**

### Grounding?

**YES.**

### Temporal change?

**YES.**

### Optical-SAR?

**YES.**

### Agentic orchestration?

**YES.**

### Natural-language understanding?

**YES. MiniLM adapter.**

### Deterministic execution?

**YES.**

### Evidence?

**YES.**

### Confidence?

**YES.**

### Execution trace?

**YES.**

### Hidden ISRO/SAC compatibility?

**CONDITIONAL.**

The actual hidden distribution remains unavailable, so compatibility can be engineered but not empirically proven before official evaluation. The supplied contract explicitly says hidden annotations are not disclosed.


# 76. Final Master Specification

## Final models

```
`Router:`

`sentence-transformers/all-MiniLM-L6-v2`

`+ classifier adapter`


`VLM:`

`HuggingFaceTB/SmolVLM-500M-Instruct`


`Grounding:`

`chendelong/RemoteCLIP`

`RemoteCLIP-ViT-B-32`

`+ grounding head`


`Change:`

`STANet-style Siamese model`


`Optical-SAR:`

`antofuller/CROMA`

`CROMA\_base.pt`

`+ lightweight fusion head`
```

## Final task map

```
`VQA`

`→ SmolVLM`


`Caption`

`→ SmolVLM`


`Grounding`

`→ MiniLM`

`→ RemoteCLIP`

`→ grounding head`

`→ SmolVLM explanation`


`Change`

`→ STANet`

`→ SmolVLM explanation`


`Optical-SAR`

`→ CROMA`

`→ fusion head`

`→ SmolVLM explanation`
```

## Final control map

```
`User`

` ↓`

`MiniLM`

` ↓`

`Intent JSON`

` ↓`

`Schema validation`

` ↓`

`Deterministic workflow`

` ↓`

`specialists`

` ↓`

`evidence`

` ↓`

`confidence`

` ↓`

`result`
```


# 77. IMPLEMENTATION MODEL HANDOFF

## Immutable architecture decisions

The implementation model must **not redesign**:

1. MiniLM intent router.

2. Deterministic policy controller.

3. Modular specialist interface.

4. SmolVLM main VLM.

5. RemoteCLIP grounding specialist.

6. STANet-style change detector.

7. CROMA-based optical-SAR path.

8. Common evidence engine.

9. Calibrated confidence.

10. Unified result schema.

11. Execution trace.

12. Dataset leakage protections.

13. Central configuration.

14. Competition-scale Gradio deployment.


## Variables the implementation model may tune

```
`router adapter width`

`router threshold`

`LoRA rank`

`VLM LR`

`tile size`

`tile overlap`

`top-K tiles`

`grounding head width`

`grounding threshold`

`change threshold`

`component filtering`

`fusion head width`

`confidence calibration`
```


## Variables that must be experimentally optimized

```
`router threshold`

`grounding learning rate`

`grounding loss weights`

`change threshold`

`change loss weights`

`VLM learning rate`

`LoRA rank`

`CROMA fusion projection width`

`tile top-K`
```

All selection must happen on validation data.


## Unknowns

### Exact official metrics

**UNVERIFIED**

Insert official definitions when published.

### Exact hidden sensor band arrangement

**UNVERIFIED**

Resolve from official evaluation input metadata only.

### Exact hidden task output format

**UNVERIFIED**

Resolve through the supplied evaluation specification when provided.

### Exact HF quota at judging time

**VARIABLE**

CPU-compatible execution remains mandatory.


# 78. Non-Negotiable Requirements

```
`\[ \] Natural language query understanding`

`\[ \] Learned NLP router`

`\[ \] Deterministic execution`

`\[ \] Single-image VQA`

`\[ \] Captioning`

`\[ \] Grounding`

`\[ \] Temporal change`

`\[ \] Optical-SAR analysis`

`\[ \] GeoTIFF`

`\[ \] Geo metadata`

`\[ \] Evidence`

`\[ \] Confidence`

`\[ \] Execution trace`

`\[ \] Remote-sensing adaptation`

`\[ \] Leakage prevention`

`\[ \] Public-test isolation`

`\[ \] Hidden-test compatible preprocessing`

`\[ \] Reproducible evaluation`

`\[ \] HF deployment`

`\[ \] Kaggle-compatible training`
```


# 79. Completion Checklist

```
`ARCHITECTURE`

`\[ \] Specialist interface implemented`

`\[ \] Controller implemented`

`\[ \] Workflow DAG implemented`

`\[ \] Result schema implemented`


`ROUTER`

`\[ \] MiniLM loaded`

`\[ \] Adapter trained`

`\[ \] Hard negatives tested`

`\[ \] Confidence threshold validated`

`\[ \] Fallback routing implemented`


`VLM`

`\[ \] SmolVLM baseline`

`\[ \] BigEarthNet adaptation`

`\[ \] validation comparison`

`\[ \] frozen prompt set`


`GROUNDING`

`\[ \] RemoteCLIP loaded`

`\[ \] VRSBench parser`

`\[ \] grounding head`

`\[ \] box conversion`

`\[ \] NMS`

`\[ \] confidence`


`CHANGE`

`\[ \] STANet-style model`

`\[ \] LEVIR-CD training`

`\[ \] change map`

`\[ \] connected components`

`\[ \] CDVQA reasoning`


`OPTICAL-SAR`

`\[ \] CROMA loaded`

`\[ \] sensor adapter`

`\[ \] optical preprocessing`

`\[ \] SAR preprocessing`

`\[ \] fusion head`

`\[ \] sensor-dropout testing`


`EVIDENCE`

`\[ \] image crops`

`\[ \] boxes`

`\[ \] masks`

`\[ \] change maps`

`\[ \] modality evidence`


`EVALUATION`

`\[ \] manifests`

`\[ \] leakage scans`

`\[ \] benchmark adapters`

`\[ \] metrics`

`\[ \] normalization`

`\[ \] immutable public test`

`\[ \] reproducible run manifest`


`GUI`

`\[ \] upload`

`\[ \] viewer`

`\[ \] query`

`\[ \] task display`

`\[ \] grounding overlay`

`\[ \] change overlay`

`\[ \] optical/SAR comparison`

`\[ \] confidence`

`\[ \] trace`

`\[ \] report`


`DEPLOYMENT`

`\[ \] HF Space`

`\[ \] CPU mode`

`\[ \] ZeroGPU mode`

`\[ \] lazy loading`

`\[ \] sequential request test`

`\[ \] cold-start test`
```


# 80. The final conceptual architecture

This is the version I would now **freeze as the competition architecture**:

```
`                       SATQUERY AI`

`                           │`

`                           ▼`

`                 ┌──────────────────┐`

`                 │ Tiny NLP Router  │`

`                 │ MiniLM + Adapter │`

`                 └────────┬─────────┘`

`                          │`

`                          ▼`

`                 ┌──────────────────┐`

`                 │ Policy Controller│`

`                 │ Deterministic    │`

`                 └────────┬─────────┘`

`                          │`

`       ┌──────────────────┼─────────────────────┐`

`       │                  │                     │`

`       ▼                  ▼                     ▼`

`     VQA/CAP         GROUNDING              CHANGE`

`    SmolVLM        RemoteCLIP + Head        STANet`

`       │                  │                     │`

`       └──────────────────┼─────────────────────┘`

`                          │`

`                          ▼`

`                OPTICAL + SAR`

`                      CROMA`

`                       +`

`                   Fusion Head`

`                          │`

`                          ▼`

`                ┌──────────────────┐`

`                │ Evidence Engine  │`

`                └────────┬─────────┘`

`                         ▼`

`                ┌──────────────────┐`

`                │ Confidence       │`

`                │ Calibration      │`

`                └────────┬─────────┘`

`                         ▼`

`                ┌──────────────────┐`

`                │ Result Contract  │`

`                │ Answer + Spatial │`

`                │ Evidence + Trace │`

`                └────────┬─────────┘`

`                         ▼`

`                     Gradio GUI`
```

And the scaling principle remains:

```
`NOW`


`Controller`

`   ↓`

`Python specialists`



`LATER`


`Controller`

`   ↓`

`specialist interface`

`   ↓`

`┌────────┬────────┬────────┬────────┐`

`VLM    Grounding  Change   Fusion`

`worker  worker    worker   worker`
```

Nothing fundamental changes.

That is the important architectural property we wanted.

### Final recommendation

**Yes, use the tiny NLP adapter.**

More specifically:

> **MiniLM embedding encoder + tiny multi-head intent adapter + deterministic workflow policy.**

It gives SatQuery actual natural-language understanding while keeping orchestration auditable and predictable.

And **yes, add grounding as a first-class specialist**, with RemoteCLIP providing the remote-sensing image-text alignment and a lightweight learned grounding head producing the actual spatial localization. RemoteCLIP is already explicitly built for remote-sensing vision-language alignment and provides ViT-B/32, ViT-L/14 and RN50 pretrained checkpoints. ([GitHub](https://github.com/ChenDelong1999/RemoteCLIP?utm_source=chatgpt.com))

The major current infrastructure correction is also locked in: **Kaggle planning should now assume T4×2, not P100**

This gives us a very clean progression:

```
`tiny NLP understanding`

`        +`

`deterministic agent`

`        +`

`specialist intelligence`

`        +`

`evidence`

`        +`

`confidence`

`        +`

`multimodal/temporal reasoning`

`        =`

`SatQuery AI`
```

And importantly, it remains a **competition prototype**, not a startup accidentally carrying Kubernetes on its back like a medieval peasant carrying a castle.

