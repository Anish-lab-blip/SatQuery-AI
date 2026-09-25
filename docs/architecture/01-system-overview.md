# 01 — System Overview

**Parent:** [Architecture hub](README.md) · **Status tags:** `IMPLEMENTED` · `VERIFIED` ·
`MEASURED` · `NOT RUN`

---

## 1. What the system is

SatQuery AI takes a natural-language question and one or two image assets, and returns a typed
`ResultEnvelope` containing an answer, a list of evidence items, an execution trace, and a
confidence breakdown.

It is a **router + specialists** architecture. There is no single end-to-end model that "does
satellite QA". Instead:

1. a **router** reads the question and predicts *which capability* is being requested;
2. a **controller** validates the request, dispatches exactly one specialist, and assembles the
   result;
3. the **specialist** computes the actual answer using a frozen backbone plus a small trained module;
4. an **evidence engine** aggregates, orders, deduplicates and bounds what the specialist produced;
5. a **confidence** stage attaches a calibrated number, or honestly declines to.

### 1.1 Why not one end-to-end model

| Constraint | Consequence |
|---|---|
| The system must run on **CPU** | End-to-end VLM inference at usable quality needs a GPU; small per-task modules do not. |
| Different tasks have **incompatible outputs** | `change` returns a spatial change map; `caption` returns prose; `optical_sar` returns a class distribution. One head cannot emit all three. |
| Tasks have **different data and metrics** | Each specialist is trained and evaluated on its own split with its own protocol. |
| **Truthfulness** | Per-task metrics are auditable. A single end-to-end number would hide which component failed. |

The cost of this design is that there is **no system-level accuracy number** — because there is no
single model to measure. That absence is stated rather than papered over.

## 2. The six capabilities

Declared in `configs/base.yaml` (`router.tasks`) and mirrored in the live
`GET /api/capabilities` response. All six report `available: true` in the deployed system.

| Task | Assets | Output | Backbone (frozen) | Trained module |
|---|---|---|---|---|
| `vqa` | 1 | short answer | SmolVLM-500M-Instruct | (unadapted; LoRA exists but is rejected) |
| `caption` | 1 | prose caption | SmolVLM-500M-Instruct | (unadapted) |
| `grounding` | 1 | boxes / regions | RemoteCLIP ViT-B/32 | grounding head |
| `change` | 2 (equal shape) | change map + regions | STANet-style (ResNet-18 + PAM) | change head |
| `change_vqa` | 2 | short answer | — (cached change features) | change-VQA head |
| `optical_sar` | 2 (GeoTIFF pair) | class distribution | CROMA-base | fusion head |

`Task` in `core/schemas.py` also carries `unsupported` — the router's explicit "this is not a
satellite-imagery question" class. The router's label space is therefore **6 classes**
(`router/label_space.py`: `vqa, caption, grounding, change, optical_sar, unsupported`), and
`change_vqa` is reached through the change family rather than being a separate router class.

> **Note the asymmetry.** `router/label_space.py` lists **six** task classes; the capabilities
> endpoint lists **six** tasks but a *different* six — `change_vqa` appears in capabilities and
> `unsupported` does not. This is intentional: `unsupported` is a routing outcome, not a servable
> capability. `core/schemas.py::Task` carries all seven values.

## 3. Component inventory

Every path below is real and is the authoritative location.

| Layer | Module | Responsibility |
|---|---|---|
| **Contracts** | `core/schemas.py` | the binding typed contract: `Task`, `Intent`, `Evidence`, `SpecialistResult`, `ResultEnvelope`, `ExecutionTrace`, … |
| **Config** | `core/config.py` | load, deep-merge, validate, hash the registry; `get_config()` singleton |
| **Errors** | `core/errors.py` | the error taxonomy (`ConfigError`, `ModelLoadError`, `WorkflowPlanError`, …) |
| **Planning** | `core/planner.py` | turn an `Intent` into a concrete workflow |
| **Registry** | `core/registry.py` | specialist registration / lookup |
| **Controller** | `core/controller.py` | the nine-state FSM; the only thing that dispatches |
| **Router** | `router/encoder.py` | frozen MiniLM embedding, cached |
| | `router/adapter.py` | the 5-head `IntentAdapter` (the only trainable router part) |
| | `router/classifier.py` | learned classification + confidence threshold |
| | `router/fallback.py` | deterministic lexical fallback (`lexical_route`) |
| | `router/label_space.py` | the ontology, single source of truth |
| | `router/dataset.py`, `router/train.py` | dataset generation and training |
| **Specialists** | `specialists/base.py` | the specialist interface |
| | `specialists/vqa/{model,inference,prompts}.py` | SmolVLM VQA |
| | `specialists/grounding/{remoteclip,head,inference,specialist}.py` | RemoteCLIP + head |
| | `specialists/change/{stanet,specialist,postprocess,vqa_specialist}.py` | change detection + change-VQA |
| | `specialists/optical_sar/{croma,fusion_head,inference,specialist,sensor_adapter,radiometry,prompts}.py` | CROMA fusion |
| **Evidence** | `evidence/engine.py` | aggregation: dedup → sort → renumber → cap |
| | `evidence/confidence.py` | temperature scaling, honest pass-through |
| **Inference app** | `app/space_app.py` | `build_space_app()`; the four-endpoint JSON contract |
| | `app/serving.py` | the composition root (`build_serving_controller()`) |
| | `app/deployment.py` | deployment helpers |
| **Gateway** | `gateway/` | the Render orchestrator |
| **Frontend** | `frontend/` | the static site |

## 4. The frozen-backbone strategy

Four backbones, all pinned by revision in `configs/base.yaml`, all fetched from the Hub on first use:

| Role | Repository | Revision | Why frozen |
|---|---|---|---|
| Router encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | embeddings are cached; the adapter trains on cached vectors in 0.28 s on CPU |
| VLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` | a 500M VLM cannot be fine-tuned end-to-end on CPU |
| Grounding | `chendelong/RemoteCLIP` | `bf1d8a3ccf2d` | provides the visual-language embedding space; only the head is trained |
| Optical-SAR | `antofuller/CROMA` | `0dd28e3d633b` | provides optical/SAR/joint embeddings; only the fusion head is trained |

Two consequences:

1. **The system is small.** The six trained artifacts total ~125 MiB. Everything else is public
   weights.
2. **Backbones are not redistributed.** The release publishes only the six trained modules, each
   with its backbone dependency documented.

## 5. The design vocabulary

`docs/ARCHITECTURE_FREEZE.md` section 5 assigns one verb per layer. This is not decoration — it
resolves real ambiguities about *where* a decision belongs:

| Layer | Verb | Consequence |
|---|---|---|
| Router | *understands* | its output (`Intent`) is **advisory only** — the controller decides |
| Policy engine / planner | *decides* | picks the workflow; may override the router |
| Specialists | *compute* | produce evidence; never decide routing |
| VLM | *explains* | produces prose; never produces a confidence number |
| Evidence engine | *proves* | aggregates; never re-derives a specialist's claim |

The `Intent` schema makes the first row explicit in code:

```python
class Intent(BaseModel):
    """Output of the learned router. Advisory only — the controller decides."""
```

## 6. What is deliberately absent

| Absent | Status |
|---|---|
| Database / persistence | by design — the gateway is stateless |
| Authentication / users | by design |
| Job queue | by design — inference is synchronous |
| GPU requirement | by design — CPU-first, `.to(device)` everywhere |
| Gradio GUI | `app/space_app.py` serves JSON only |
| Chain-of-thought in traces | by design — observable facts only |
| System-level end-to-end benchmark | **NOT RUN — none exists** |
| Router test-split evaluation | **NOT RUN** |

## 7. Evidence for this document

| Claim | Source |
|---|---|
| six tasks, config-declared | `configs/base.yaml` §`router.tasks` |
| router label space is 6 classes | `router/label_space.py` §`TASK_CLASSES` |
| `Intent` is advisory | `core/schemas.py` §`Intent` docstring |
| layer verbs | `docs/ARCHITECTURE_FREEZE.md` §5 (quoted in `evidence/engine.py`) |
| frozen backbones + revisions | `configs/base.yaml` |
| six trained artifacts | `release/repo/models/manifest.json` (generated) |
