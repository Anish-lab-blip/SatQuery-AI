# Development Guide

**Status tags:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED` · `NOT RUN` · `BLOCKED` ·
`DEFERRED` · `OPEN` · `RESOLVED` · `BY DESIGN`.

This is the contributor-facing guide to the SatQuery AI codebase. It answers: *what do I need to
install*; *how do I run it*; *what is each directory for*; *what rules will break the build if I break
them*; *how do I add a component*; *how do I test*; and *which environment traps will cost me an hour if
nobody tells me about them*.

Every command, path, timeout and rule below comes from a file that was read. Where the evidence does not
settle a question, the text says `UNKNOWN — not established from the available evidence` rather than
guessing.

> **Read this first.** `configs/base.yaml` is the single registry, and its hash — **`78f1e3700da15aa1`**
> — is frozen. **Editing the config moves the hash and invalidates every artifact keyed to it.** This is
> the one rule in the repository whose violation is not locally reversible (§7).

---

## Table of contents

**Part I — Prerequisites and setup**
1. Prerequisites
2. Local setup
3. Running the service locally

**Part II — Repository layout**
4. The top-level directory map
5. Where the contracts live
6. The package layout inside `core/`, `specialists/`, `app/`

**Part III — The configuration discipline**
7. "No magic numbers in Python"
8. What happens if you break the rule
9. The hash-exempt path: environment overrides
10. The invariants the loader enforces

**Part IV — Contract-first development**
11. `core/schemas.py` is the binding contract
12. The specialist interface
13. How to add a new specialist
14. How evidence is emitted

**Part V — The test workflow**
15. Running tests
16. The evidence-class markers
17. The known failures, and why they are not regressions

**Part VI — Scripts and notebooks**
18. The class of scripts under `scripts/`
19. `notebooks/`

**Part VII — Training entry points**
20. Local training (router, grounding, change, optical-SAR)
21. External GPU training (change-VQA, VLM LoRA)
22. Calibration

**Part VIII — Coding conventions**
23. What the code actually does

**Part IX — Known development traps**
24. The traps, in one table
25. Trap 1 — the stale untracked `deploy/`
26. Trap 2 — the dead sandbox proxy
27. Trap 3 — pytest only in the venv
28. Trap 4 — Chrome drops synthetic CDP key events
29. Trap 5 — Cloudflare `_headers` concatenate
30. Trap 6 — the annotation-scope trap
31. Trap 7 — the full-suite bulk-delete guard
32. Trap 8 — the stale serve process

**Part X — Status and evidence**
33. `NOT RUN` / `OPEN` / `BLOCKED` / `UNKNOWN` for development
34. Where the evidence lives

---

# Part I — Prerequisites and setup

## 1. Prerequisites

| Requirement | Value | Source |
|---|---|---|
| Python | **3.11+** | `release/repo/README.md` §Installation |
| Device | **a CPU is sufficient**; no CUDA requirement | `release/repo/README.md` §Installation |
| GPU | not required | `docs/DEPLOYMENT_DECISION.md` §5 |
| Git | needed to clone | `release/repo/README.md` §Installation |

The codebase is CPU-first and the deployment is CPU-only. This is not a fallback — it is the design:

- `core/config.py:86-91` — `device_preference` honours the `SATQUERY_DEVICE` env override, else
  `"cuda" if torch.cuda.is_available() else "cpu"`.
- Every specialist defaults to `device="cpu"` (`docs/DEPLOYMENT_DECISION.md` §5).
- **No `.cuda()` call exists anywhere**; all placement is `.to(device)` (`docs/DEPLOYMENT_DECISION.md`
  §5).
- `configs/base.yaml:293` — `cpu_mode_required: true`.

The development container that runs the inference tier pins Python 3.12
(`.devcontainer/devcontainer.json:3`, `image: mcr.microsoft.com/devcontainers/python:3.12`). The
repository's own guidance is 3.11+; a 3.11 or 3.12 interpreter both work.

**Not implemented** (noted, not needed): no thread capping (`torch.set_num_threads`) and no
quantisation. `lazy_load` and `cache_max_models` are only *reported* (`app/deployment.py:609-610`,
`:896-897`); **UNVERIFIED** whether any code enforces a one-model cache (`docs/DEPLOYMENT_DECISION.md`
§5).

## 2. Local setup

```bash
git clone https://github.com/Anish-lab-blip/SatQuery-AI
cd SatQuery-AI
python -m venv .venv
source .venv/Scripts/activate      # Windows git-bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

(`release/repo/README.md` §Installation)

### 2.1 The dependency manifest and its two profiles

`requirements.txt` is **frozen at architecture v1.0** and declares two install profiles in its header
comment (`requirements.txt:1-8`):

```
# Two install profiles:
#   CPU (local dev / schema / geo / unit tests):
#       pip install -r requirements.txt
#   GPU (Kaggle T4x2 / HF ZeroGPU): torch is preinstalled on both.
#       Do NOT pin torch here — Kaggle and HF ship their own builds.
```

**torch is deliberately not pinned** (`requirements.txt:21`). The platform supplies it. This is why the
local CPU install works on a machine with no CUDA and why the Kaggle/ZeroGPU targets get their own
builds.

The manifest's sections, with the contract notes the file itself carries:

| Section | Packages | Note |
|---|---|---|
| core | `numpy`, `pyyaml`, `pydantic` | — |
| geospatial | `rasterio`, `pyproj`, `opencv-python-headless` | — |
| models | `transformers>=4.52`, `open-clip-torch>=2.24`, `sentence-transformers>=2.7`, `peft>=0.10`, `huggingface_hub>=0.23`, `safetensors>=0.4`, `einops>=0.7` | see below |
| ui + reporting | `gradio>=4.44`, `reportlab>=4.1` | — |
| dev | `pytest>=8.0`, `pytest-cov>=5.0` | — |

Three contract notes in the file are load-bearing (`requirements.txt:43-53`):

- `transformers>=4.52` — SmolVLM via `AutoModelForImageTextToText`; `AutoModelForVision2Seq` is
  deprecated (finding C-2).
- `open-clip-torch` — the RemoteCLIP checkpoint is loaded via `pretrained=<path>` so
  `load_checkpoint()` runs its state-dict fixups (C-4).
- `huggingface_hub` — used to pin the RemoteCLIP / SmolVLM / CROMA revisions.

> **The `einops` lesson.** `einops>=0.7` is **not optional**. It is required by the vendored
> `specialists/optical_sar/vendor/use_croma.py` (`from einops import rearrange`). It was found missing on
> 2026-09-18 by executing the vendored module: it raised `ModuleNotFoundError`, so the CROMA path was
> blocked on a dependency that no document declared (`requirements.txt:28-33`).

### 2.2 Device selection

Set `SATQUERY_DEVICE` to force a device:

```bash
export SATQUERY_DEVICE=cpu        # cpu | cuda | mps | null
```

`device_preference` reads the env var first and only falls back to a torch probe if it is unset
(`core/config.py:87-91`). The value is read **without importing torch** on the metadata path, which is
what keeps the health route cheap (`docs/DEPLOYMENT_TOPOLOGY.md` §3.3).

## 3. Running the service locally

The inference service is served by the same launcher the Codespace runs
(`release/repo/README.md` §Local development):

```bash
# Inference service, CPU (this is the launcher the Codespace runs)
PORT=8000 python deploy/codespace/serve.py

# Health
curl localhost:8000/v1/health
```

`serve.py` builds the app through `build_space_app()` and binds it with uvicorn on `$PORT` (default 8000)
(`deploy/codespace/serve.py:16-25`). It imports cheaply — FastAPI is imported inside `build_space_app()`
and no model is loaded at module scope — so it stays import-safe on a CPU host with no GPU and no weights
present (`deploy/codespace/serve.py:1-13`).

The server answers four routes (`app/space_app.py`):

| Route | Line | Purpose |
|---|---|---|
| `GET /v1/health` | `app/space_app.py:521` | liveness, derived status, torch-free device probe |
| `GET /v1/capabilities` | `app/space_app.py:549` | the six declared capabilities |
| `POST /v1/assets` | `app/space_app.py:555` | handle-based upload (off unless enabled) |
| `POST /v1/analyze` | `app/space_app.py:661` | the analysis path |

`POST /v1/assets` is **off unless explicitly enabled** (`SATQUERY_ASSET_ENABLED` **and**
`SATQUERY_ASSET_DIR` must both be set; otherwise it answers `503` rather than defaulting to a temp
directory) (`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §3.1.1).

### 3.1 Serving the frontend locally

The frontend is fully static and needs no build step (`release/repo/README.md` §Local development):

```bash
python -m http.server 5500 --directory frontend
```

The gateway's dev-origin allowlist includes `localhost` and `127.0.0.1` on ports 3000, 5500, 5173, 8000
and 8080 (`deploy/render/main.py:139-143`), so a local dev server on any of those ports is accepted
without an env-var change.

### 3.2 Running the gateway locally

```bash
PORT=10000 \
GITHUB_TOKEN=<token> \
CODESPACE_NAME=<codespace> \
SATQUERY_ALLOWED_ORIGINS="https://satquery.pages.dev" \
uvicorn deploy.render.main:app --port 10000
```

(`deploy/render/README.md` §Running locally — the placeholder is the repository's own.)

Then exercise it:

```bash
curl http://localhost:10000/api/health
curl -X POST http://localhost:10000/api/infer -H 'content-type: application/json' -d '{"query":"..."}'
curl http://localhost:10000/api/capabilities
```

> **The monorepo `deploy/` is stale** (§25). Use it for local experimentation only; it is **not** the
> deployed source and it lacks the tunnel code the live service runs.

---

# Part II — Repository layout

## 4. The top-level directory map

| Directory / file | Purpose | Source |
|---|---|---|
| `app/` | the serving composition root and the FastAPI entrypoint | `app/serving.py`, `app/space_app.py`, `app/deployment.py` |
| `core/` | the config loader, the typed schemas, the controller, the planner, the registry, the error taxonomy | `core/config.py`, `core/schemas.py`, `core/controller.py`, `core/planner.py`, `core/registry.py`, `core/errors.py`, `core/code_revision.py` |
| `router/` | the MiniLM intent router: encoder, classifier, adapter, training, label space, lexical fallback | `router/encoder.py`, `router/classifier.py`, `router/adapter.py`, `router/train.py`, `router/dataset.py`, `router/fallback.py`, `router/label_space.py` |
| `specialists/` | the six specialist implementations behind one interface | `specialists/base.py`, `specialists/vqa/`, `specialists/grounding/`, `specialists/change/`, `specialists/optical_sar/` |
| `preprocessing/` | raster loading, imagery, quality checks | `preprocessing/raster.py`, `preprocessing/imagery.py`, `preprocessing/quality.py` |
| `geospatial/` | CRS handling and transforms | `geospatial/crs.py`, `geospatial/transform.py` |
| `evidence/` | the evidence engine and confidence | `evidence/engine.py`, `evidence/confidence.py` |
| `evaluation/` | manifests, leakage, metrics, normalisation, the runner, benchmark adapters, the frozen prompt set | `evaluation/manifests.py`, `evaluation/leakage.py`, `evaluation/metrics/`, `evaluation/normalize.py`, `evaluation/runner.py`, `evaluation/benchmark_adapters/`, `evaluation/prompt_freeze.json`, `evaluation/manifest_freeze.json` |
| `training/` | the training loops and data adapters, split by task | `training/router/`, `training/grounding/`, `training/change/`, `training/change_vqa/`, `training/fusion/`, `training/vlm/`, `training/calibration/`, `training/data/` |
| `configs/` | the frozen registry and the frozen deploy manifest | `configs/base.yaml`, `configs/deploy.yaml` |
| `gateway/` | the standalone gateway: policy (decisions) + app (HTTP plumbing) | `gateway/policy.py`, `gateway/app.py`, `gateway/assets.py` |
| `deploy/` | the deployment launchers — **stale and untracked in the monorepo** | `deploy/codespace/`, `deploy/render/` (§25) |
| `tests/` | the suites, split by concern and evidence class | `tests/unit/`, `tests/integration/`, `tests/routing/`, `tests/geospatial/`, `tests/leakage/`, `tests/model/`, `tests/e2e/` |
| `scripts/` | the flat set of executable helpers | §18 |
| `notebooks/` | the Kaggle notebooks | §19 |
| `artifacts/` | trained heads, checkpoints, caches, evidence archives | ~3.7 GB total (`release/CURRENT_RELEASE_STATE.md` §3) |
| `frontend/` | the static site | staged by `scripts/stage_pages.mjs` |
| `docs/` | the project's own engineering records | ~60 files |
| `hf/` | the Hugging Face project card / model cards | — |
| `demo/`, `benchmark/`, `reports/`, `data/`, `logs/` | supporting material | — |

> **`training/router/` is empty** in the working copy; the router's training lives in `router/train.py`
> and `router/dataset.py` and is driven by `scripts/train_router.py` (§20).

> **The repository has no `pyproject.toml` and no `setup.py`.** `app` is a plain package, so the repo root
> must be on `sys.path` for `from app.space_app import ...` to resolve
> (`deploy/codespace/launch.sh:31-34`; `pytest.ini` sets `pythonpath = .`).

## 5. Where the contracts live

Three files are authoritative, and the code — not a document — wins when they disagree:

| Contract | File | What it fixes |
|---|---|---|
| the config registry | `configs/base.yaml` | every tunable value; hashed (§7) |
| the wire/data schemas | `core/schemas.py` | the request, result, evidence, trace and health shapes |
| the specialist interface | `specialists/base.py` | the four-method contract every specialist implements |
| the capability table | `core/registry.py` | which capabilities exist and how to build them |
| the planner's mapping | `core/planner.py` | `TASK_CAPABILITY` and `CAPABILITY_ASSETS` |

When `docs/API_CONTRACT.md` and `core/schemas.py` disagree, the contract document's own authority clause
resolves it: *"the request/response shapes are not invented here. They are the existing, tested Pydantic
models in `core/schemas.py`. This document describes them; it does not declare new ones."*
(`docs/STEP7_BACKEND_CHAIN_REPORT.md` §15). The known instance of this — the forward-compatibility
promise versus `extra="forbid"` — is recorded as **C-2** and left as a documented contradiction rather
than silently resolved (`docs/STEP7_BACKEND_CHAIN_REPORT.md` §15).

## 6. The package layout inside `core/`, `specialists/`, `app/`

### 6.1 `core/`

| File | Role |
|---|---|
| `core/config.py` | the loader, the validator, the hash (§7, §10) |
| `core/schemas.py` | every Pydantic model (§11) |
| `core/controller.py` | the nine-state controller that runs a plan and produces the trace |
| `core/planner.py` | the policy layer: `TASK_CAPABILITY`, `CAPABILITY_ASSETS`, `plan()` |
| `core/registry.py` | the capability registry: spec table, lazy construction, degradation states |
| `core/errors.py` | the typed error taxonomy and `scrub_paths` |
| `core/code_revision.py` | the code revision recorded in a run |

The three-tier control split is deliberate: the **registry** knows *which* specialists exist and *how* to
construct them; the **planner** *decides* what runs; the **controller** executes. The registry's own
docstring states it: *"It never decides what runs — that is `core.planner`'s job alone"*
(`core/registry.py:6-8`).

### 6.2 `specialists/`

| Package | Files | Capability |
|---|---|---|
| `specialists/vqa/` | `inference.py`, `model.py`, `prompts.py` | `vqa`, `caption` |
| `specialists/grounding/` | `specialist.py`, `remoteclip.py`, `head.py`, `inference.py` | `grounding` |
| `specialists/change/` | `specialist.py`, `stanet.py`, `vqa_specialist.py`, `postprocess.py` | `change`, `change_vqa` |
| `specialists/optical_sar/` | `specialist.py`, `croma.py`, `fusion_head.py`, `sensor_adapter.py`, `radiometry.py`, `inference.py`, `prompts.py`, `vendor/` | `optical_sar` |

### 6.3 `app/`

| File | Role |
|---|---|
| `app/serving.py` | the composition root — wires checkpoints through the registry's `builders=` override |
| `app/space_app.py` | `build_space_app()` — the FastAPI app and the four routes |
| `app/deployment.py` | the deployment description and device resolution |

---

# Part III — The configuration discipline

## 7. "No magic numbers in Python"

`configs/base.yaml` opens with the rule, in the file itself:

```yaml
# RULE: no magic numbers anywhere in Python. Everything tunable lives here.
# Every value below is loaded, validated and hashed by core/config.py.
```

(`configs/base.yaml:4-5`)

The loader reinforces it: *"One config system. No duplicated constants. Every value in configs/base.yaml
is loaded, validated against the frozen architecture, and hashed so evaluation runs are reproducible."*
(`core/config.py:1-5`).

**Access pattern.** Never read the YAML directly; import the singleton:

```python
from core.config import get_config

cfg = get_config()
cfg.get("croma.image_resolution")     # dotted-path access
cfg.require("change.encoder")         # raises ConfigError if missing
cfg.seed                              # project.seed, default 42
```

`get_config()` is an `lru_cache(maxsize=1)` singleton — *"Import this, do not re-read YAML"*
(`core/config.py:270-273`).

## 8. What happens if you break the rule

Two distinct failures, and both are loud.

### 8.1 The loader fails startup

`Config.__init__` calls `_validate()`, which collects **every** violation and raises a single
`ConfigError` naming them all (`core/config.py:46-49,94-222`):

```python
if errors:
    raise ConfigError(
        "configuration failed frozen-architecture validation:\n  - "
        + "\n  - ".join(errors)
    )
```

The header states the intent: *"if a config tries to violate a frozen decision (e.g. bf16 on T4,
torch.compile on ZeroGPU, a CROMA image_resolution that is not a multiple of 8), it fails loudly rather
than at runtime"* (`core/config.py:6-8`).

### 8.2 Editing the config MOVES the hash

The hash is a sha256 over the whole registry, truncated to 16 hex characters
(`core/config.py:76-80`):

```python
@property
def hash(self) -> str:
    """Stable hash of the whole registry. Recorded in every evaluation run."""
    blob = json.dumps(self._data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
```

Because it is computed over the entire registry, **any** edit to `configs/base.yaml` changes it. The
frozen value is **`78f1e3700da15aa1`** (`release/repo/README.md` §Reproducibility;
`docs/PHASE18_DEPLOYMENT_PACKAGING.md` §3). Every artifact records the hash it was produced against, so a
hash change **invalidates every artifact keyed to it**.

> **This is the one non-reversible action in the repository.** `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §6.1
> lists it as the single row in the rollback table whose answer to "Reversible?" is **no**: *"The recorded
> benchmark hash is gone; the frozen benchmark no longer matches."*

**Verify the hash after any config-adjacent change:**

```bash
$PY -c "from core.config import get_config; print(get_config().hash)"
# expected: 78f1e3700da15aa1
```

(`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §1.1, where `$PY` is `$REPO/.venv/Scripts/python.exe` on Windows.)

**Two things that do *not* move the hash** — both verified:

1. **`configs/deploy.yaml` is inert.** It carries a `registry: false` marker and is never merged into the
   registry, so editing it cannot move the hash. Verify with `$PY scripts/validate_deploy_config.py` →
   exit 0, `"deploy.yaml is inert (not in the registry) and C-8-consistent."`
   (`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §2.4). **But** `scripts/validate_deploy_config.py` hard-fails if
   the `deployment:` block in `deploy.yaml` differs key-for-key from `base.yaml`'s, so a change to one
   alone fails the validator (`docs/DEPLOYMENT_DECISION.md` §4).
2. **The serving wiring uses `builders=`.** `app/serving.py` wires the change and change-VQA heads through
   the registry's `builders=` override instead of config, which is *"the seam that keeps `Config.hash`
   unchanged while still pointing serving at the trained artifacts"* (`docs/PHASE19_FINAL_HARDENING.md`
   §3.4).

## 9. The hash-exempt path: environment overrides

Two registry values can be overridden from the environment without editing the YAML
(`core/config.py:261-265`):

| Variable | Effect |
|---|---|
| `SATQUERY_PRECISION` | overrides `training.precision` |
| `SATQUERY_TORCH_COMPILE` | overrides `deployment.torch_compile` (`"true"` → `True`) |

**Both are still validated.** Setting `SATQUERY_TORCH_COMPILE=true` **fails startup** because finding C-8
forbids `torch.compile` (`release/repo/docs/DEPLOYMENT.md` §6.3; `core/config.py:114-119`).

This is the general pattern for deployment state that must not move the hash: read it from the
environment. The asset-store capacity, TTL and per-file cap follow the same rule
(`release/repo/README.md` §Installation).

## 10. The invariants the loader enforces

`core/config.py::_validate` is not documentation — it is a check that raises `ConfigError`. Each invariant
exists because a specific finding proved the failure mode.

| Invariant | Why it exists | Finding |
|---|---|---|
| `croma.image_resolution % 8 == 0` | CROMA asserts this; native 120 → 225 patches | C-7 |
| `training.precision ∈ {fp16, bf16, fp32}` | the T4 is SM 7.5, so bf16 is unavailable | C-6 |
| `deployment.torch_compile is not true` | ZeroGPU does not support `torch.compile` | C-8 |
| `vlm.processor_longest_edge ≤ image.tile_size` | the processor's default `longest_edge` is 2048, which upscales a 512 px tile 4× and then splits it into **17** sub-images — a ~17× overrun, not the 4× the plan estimated | F5-2 |
| `vlm.prompt_must_use_chat_template is true` | SmolVLM raises `ValueError` on prompts lacking one `<image>` token per image | F5-3 |
| `fusion.input_dim == 3*encoder_dim + optical_channels + sar_channels` (= 2318) | CROMA emits optical/SAR/joint GAP vectors; the availability mask is consumed by the head | C-1 |
| `croma.optical_channels == 12` and `croma.sar_channels == 2` | CROMA's `s2_channels` / `s1_channels` are fixed | — |
| `grounding_head.feature_dim == 4 * grounding.encoder_projected_dim` (= 2048) | a mismatch is a **silent** shape error — torch raises only at the similarity step, after patch features are already cached | P7-1 |
| `router.tasks` includes `unsupported` and `router.num_tasks == len(router.tasks)` | the ontology and its declared size cannot drift apart | — |
| `change.sa_mode ∈ {BAM, PAM}` and `change.encoder` is set | the change architecture is not implicit | C-9 |
| `image.top_k_tiles ≤ image.max_tiles` | the dispatch ceiling cannot exceed the examination ceiling | — |

(`core/config.py:94-222`; `release/repo/README.md` §Installation.)

> **Why the grounding-head guard matters most.** A mismatch there is a *silent* shape error: torch raises
> only at the similarity step, by which point the patch features have already been computed and cached —
> *"the failure surfaces far from its cause"* (`core/config.py:170-196`).

---

# Part IV — Contract-first development

## 11. `core/schemas.py` is the binding contract

`core/schemas.py` (462 lines) holds every model the system exchanges. The rule is simple: **no specialist
may invent its own result shape.** Every specialist returns exactly `SpecialistResult`
(`core/schemas.py:325-406`), whose docstring says so: *"Every specialist returns exactly this. No
exceptions."*

The models, in order:

| Model | Line | Role |
|---|---|---|
| `Task` (enum) | 35 | the six-task ontology (`vqa`, `caption`, `grounding`, `change`, `optical_sar`, `change_vqa`) + `unsupported` |
| `Modality` (enum) | 50 | `optical` / `sar` / `joint` |
| `CoordinateSystem` (enum) | 57 | `normalized_0_1` / `geo` |
| `EvidenceType` (enum) | 65 | the evidence classes |
| `ControllerState` (enum) | 79 | the nine-state spine |
| `Intent` | 94 | the router's output |
| `GeoMetadata` | 120 | CRS / transform metadata |
| `SensorDescriptor` | 138 | sensor identity |
| `AssetMetadata` | 153 | one input asset |
| `Box` | 171 | a flat box (see the flat-vs-nested trap below) |
| `Region` / `ChangeRegion` | 189 / 202 | spatial outputs |
| `Evidence` | 216 | one observable artefact |
| `ConfidenceBreakdown` | 259 | measurable confidence |
| `TraceStep` / `ModelRef` / `ExecutionTrace` | 279 / 288 / 296 | the trace |
| `SpecialistResult` | 325 | the master contract |
| `AnalysisRequest` | 412 | the request |
| `ResultEnvelope` | 421 | the response wrapper |
| `HealthStatus` | 430 | the health shape |

**Every model sets `model_config = ConfigDict(extra="forbid")`.** This is deliberate and has a documented
consequence (C-2, §5): the models reject a body carrying an unknown key, which contradicts
`API_CONTRACT.md` §1.1's forward-compatibility promise. The code is authoritative; the document is the
inaccurate half (`docs/STEP7_BACKEND_CHAIN_REPORT.md` §15).

> **The flat-vs-nested trap.** `Box` is **flat**, not nested. An earlier documentation draft described it
> with nested geometry, and *"every box the frontend drew would have been at the origin"*
> (`docs/PHASE19_FINAL_HARDENING.md` §3.6). Documentation is validated against the real models by tests
> (`test_api_contract_doc.py`, `test_frontend_guide_doc.py`), which is what caught it.

### 11.1 The two cross-field validators you must not break

`SpecialistResult` carries a `_task_output_consistency` validator (`core/schemas.py:353-406`) with two
rules:

- **Grounding with no localisation is degraded, not a crash.** If `task == GROUNDING` and there are no
  boxes or regions, the validator appends a warning and sets `degraded = True`.
- **Change-VQA with no answer text is degraded.** If `task == CHANGE_VQA` and the answer is blank, the
  validator marks it degraded — but only if the specialist has not already set `degraded` itself.

The `CHANGE` clause that once lived there was **removed** (F-16c): it had collapsed to `not regions`, and
a *successful* no-change analysis began reporting `degraded: true`. **CHANGE is the one task whose
`degraded` flag is now set entirely by its specialist** (`core/schemas.py:362-395`).

## 12. The specialist interface

Every specialist implements exactly one abstract base class, `Specialist`, in `specialists/base.py`. The
interface is **four methods**, and the split is deliberate (`specialists/base.py:1-17`):

```
validate_request    -> can this specialist serve this request at all?
execute             -> do the work, return a SpecialistResult
produce_evidence    -> what observable artefacts support the result?
estimate_confidence -> what measurable signals support the score?
```

The class attributes a specialist must set (`specialists/base.py:48-58`):

| Attribute | Meaning |
|---|---|
| `name` | stable machine-readable name, used in traces and evidence sources |
| `version` | semantic version; *"bump when behaviour changes, not when code moves"* |
| `capabilities` | the capability strings this specialist serves, e.g. `("vqa", "caption")` |

The four abstract methods (`specialists/base.py:62-96`):

| Method | Contract |
|---|---|
| `validate_request(request)` | raise the **most specific** typed error available (`InvalidRequestError`, `PairMisalignmentError`, …), never a bare `Exception` |
| `execute(request)` | return a normalised `SpecialistResult`; *"Never returns None."* |
| `produce_evidence(result)` | return `list[Evidence]`; *"Must not invent anything the specialist did not actually compute."* |
| `estimate_confidence(result)` | return a `ConfidenceBreakdown`; *"Never an LLM utterance."* |

The base class also provides helpers a specialist should use rather than re-implement
(`specialists/base.py:98-134`): `supports()`, `require_assets()` (asserts an exact asset count and raises
a typed error), `require_capability()`, `model_refs()` (for the trace), and `describe()`.

`SpecialistRequest` (`specialists/base.py:34-45`) is the input: `assets`, `query`, `params`, `run_id`.
Its docstring states the isolation rule: *"Everything a specialist is given. No specialist reads global
state."*

## 13. How to add a new specialist

This is the verified procedure, read from the code. There are **five** steps, and skipping any one fails
loudly (which is the design).

### Step 1 — implement the `Specialist` ABC

Subclass `specialists.base.Specialist`, set `name`, `version` and `capabilities`, and implement the four
methods. The builder for the class is a module-level function
`build_<x>_specialist(config, *, device, **kwargs)` — the same shape as the four existing builders
(`core/registry.py:185-259` names them: `build_vqa_specialist`, `build_grounding_specialist`,
`build_change_specialist`, `build_change_vqa_specialist`, `build_optical_sar_specialist`).

### Step 2 — add a row to the spec table

Add a `SpecialistSpec` to `default_specs()` in `core/registry.py:171-260`. The spec's fields
(`core/registry.py:121-165`):

| Field | Meaning |
|---|---|
| `name` | the registry key — **the capability string**, never the specialist's own `name` |
| `capabilities` | the tuple the built object must declare; **asserted** after construction |
| `module` | dotted module path containing the builder |
| `builder` | builder function name inside `module` |
| `requires_assets` | exact asset count, or `None` for "any" |
| `config_keys` | `{builder_kwarg: dotted.config.key}` — only keys that resolve are passed |
| `optional_config_keys` | as above, but a missing key contributes nothing |
| `failure_states` | registry state to use when construction raises, keyed by exception class name |

The `SpecialistSpec.__post_init__` refuses a spec whose `name` is not in its own `capabilities`
(`core/registry.py:158-165`): *"the registry keys on capability, so this spec would be unreachable."*

### Step 3 — understand the capability-vs-name asymmetry

**This is the trap the registry exists to encode** (`core/registry.py:30-45`). The VQA specialist declares:

```python
name = "vlm"                       # specialists/vqa/inference.py:114
capabilities = ("vqa", "caption")  # specialists/vqa/inference.py:116
```

Every other specialist's `name` equals its single capability. A registry keyed on `name` would make `vqa`
permanently unfindable while every other specialist kept working. So the registry keys on **capability**
and **asserts** the capability tuple against the constructed object
(`core/registry.py:497-515`). If your spec's `capabilities` disagrees with what your specialist declares,
construction fails with a `SpecialistError` naming both tuples.

### Step 4 — register the task (only if it is a *new* task)

If the new specialist serves an existing task, nothing else is needed. If it is a new task, add it to:

- `Task` in `core/schemas.py:35-49`, and
- `TASK_CAPABILITY` in `core/planner.py:121-128`, and
- `CAPABILITY_ASSETS` in `core/planner.py:133-142`.

`CAPABILITY_ASSETS` mirrors each specialist's own `validate_request`, which **stays authoritative** — the
planner's copy is a *"cheap precondition so it can refuse before construction is attempted"*
(`core/planner.py:130-132`). A test asserts `CAPABILITY_ASSETS` equals `SpecialistSpec.requires_assets`
in both directions, and asserts the gateway carries **no third copy** of the asset-count table
(`docs/STEP7_BACKEND_CHAIN_REPORT.md` §3).

> **Adding a task touches `router.tasks`.** `router.num_tasks` must equal `len(router.tasks)`
> (`core/config.py:199-206`), so a new task means a config edit — **which moves the hash** (§8.2). This is
> the one part of "add a specialist" that has a global consequence.

### Step 5 — return the right shapes

`execute` must return a `SpecialistResult`; `produce_evidence` must return `list[Evidence]`;
`estimate_confidence` must return a `ConfidenceBreakdown`. The registry will construct your specialist
lazily and record its state.

### What the registry does with your specialist

| Behaviour | Mechanism |
|---|---|
| lazy construction | the builder is imported via `importlib.import_module` **inside** `build()` — no module-level specialist import (`core/registry.py:15-25,462-475`) |
| memoisation | the second `build(cap)` returns the cached entry (`core/registry.py:432-434`) |
| three states | `AVAILABLE` / `DEGRADED` / `UNAVAILABLE` (`core/registry.py:102-107`) |
| degradation detection | duck-typed on `has_checkpoint` / `has_head` / `has_encoder` / `model is None` (`core/registry.py:527-548`) |
| failure is retained | a construction failure returns an `UNAVAILABLE` entry rather than raising; **only an unknown capability raises** (`core/registry.py:418-450`) |
| corrupt ≠ missing | `ModelLoadError` / `ModelUnavailableError` map to `UNAVAILABLE`, never retried as `DEGRADED` (`core/registry.py:550-612`) |
| path scrubbing | the client-visible `detail` is `scrub_paths(...)`; the raw string goes to the log only (`core/registry.py:560-598`) |

> **The corrupt-vs-missing rule is load-bearing.** *"silently running an untrained model because a real
> checkpoint failed to load would be the worst outcome"* (`specialists/change/specialist.py:834-838`,
> quoted in `core/registry.py:63-75`). Do not add a fallback that re-adds a degradation the builder
> refused.

## 14. How evidence is emitted

Evidence is produced by `Specialist.produce_evidence(result)` and returned as `list[Evidence]`. The
`Evidence` model (`core/schemas.py:216-256`):

| Field | Type | Note |
|---|---|---|
| `evidence_id` | `str` | auto-generated `ev_<hex>`; **must be unique within a result** |
| `type` | `EvidenceType` | the evidence class |
| `source_specialist` | `str` | which specialist computed it |
| `coordinate_system` | `CoordinateSystem \| None` | required for spatial evidence |
| `coordinates` | `list[float] \| None` | the geometry |
| `score` | `float \| None` | `0.0 ≤ score ≤ 1.0` |
| `artifact_ref` | `str \| None` | **never a filesystem path** (F-16) |
| `payload` | `dict[str, Any]` | structured detail |

Two validators enforce correctness:

- `SpecialistResult._unique_evidence_ids` rejects a result with duplicate `evidence_id` values
  (`core/schemas.py:345-351`).
- `Evidence._spatial_needs_crs` rejects spatial evidence (`BOUNDING_BOX`, `MASK`, `CHANGE_MAP`, `TILE`,
  `IMAGE_CROP`, `JOINT_FEATURE_REGION`) that carries coordinates but **no** `coordinate_system`
  (`core/schemas.py:241-255`).

> **`artifact_ref` is never a filesystem path.** v1 exposes no artifact-serving endpoint, so it is null
> unless a deployment supplies a client-fetchable reference. *"An artifact may still be written
> server-side where configured; being written is not the same as being retrievable."*
> (`core/schemas.py:227-238`.)

---

# Part V — The test workflow

## 15. Running tests

### 15.1 Always use the repository venv interpreter

pytest is installed **only** in the repository virtualenv. Invoking the system `pytest` fails or resolves
to a different interpreter (`release/repo/docs/REPRODUCIBILITY.md` §10.2):

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q
```

On Windows, pytest must be given `-p no:cacheprovider` because the sandbox refuses `.pytest_cache` writes
(`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §0).

### 15.2 Run targeted files, not the whole tree

**A full `tests/unit` run trips the sandbox's bulk-delete guard** (4× `test_safe_delete_shim` failures)
(`release/repo/docs/REPRODUCIBILITY.md` §10.3). The workaround is to run the targeted suites:

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q     # 106 passed
```

The runbook notes that **multi-suite invocations in one command have been refused by the environment
before; single suites are reliable** (`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §2.6):

```bash
$PY -m pytest tests/unit/test_deploy_config.py -p no:cacheprovider -q
$PY -m pytest tests/unit/test_app_serving.py -p no:cacheprovider -q
$PY -m pytest tests/unit/test_api_contract_doc.py tests/unit/test_frontend_guide_doc.py -p no:cacheprovider -q
```

> **Do not pipe pytest through `grep`** in the authoring sandbox — output is block-buffered and a killed
> pipeline swallows it. Redirect to a file instead (`release/repo/docs/REPRODUCIBILITY.md` §5.6).

### 15.3 `pytest.ini`

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q --tb=short
```

(`pytest.ini:1-4`)

`pythonpath = .` is what makes `from core.config import ...` resolve from the repo root without an
installed package.

## 16. The evidence-class markers

`pytest.ini` registers four markers whose purpose is to make a test's **evidence class** explicit
(`pytest.ini:23-27`):

| Marker | Meaning |
|---|---|
| `unit` | fast, no I/O, no server, no network. Evidence about a component in isolation. |
| `integration` | exercises two or more real components wired together in-process. |
| `smoke` | a minimal end-to-end path run against real local artifacts, not a mock. |
| `real_inference` | ran the real model on real inputs in this environment. |

The file's comment states the discipline:

> *"a unit test never claims a deployment was exercised, and no test may be reported as 'real inference'
> unless it ran the real model on real data in this environment."* (`pytest.ini:13-17`)

**`environment_blocked` is deliberately not a marker.** *"a blocked path is reported in the STEP 7 report
rather than encoded as a permanently-skipped test, because a skip can be mistaken for coverage"*
(`pytest.ini:19-21`). An unregistered mark is an **error** rather than a silent no-op (`pytest.ini:14-15`).

## 17. The known failures, and why they are not regressions

Running the entire unit tree trips **5–6** failures, classified by cause
(`release/repo/docs/REPRODUCIBILITY.md` §5.4):

| # | Failure | Cause | Regression? |
|---|---|---|---|
| 1–4 | `test_safe_delete_shim` (×4) | the sandbox's bulk-**delete guard** (Windows verbatim-path behaviour) | **No** |
| 5 | one ordering flake in the router route test | passes in isolation; order/collection-dependent | **No** |
| 6 | one stale adapter test | asserts `optical_sar` absent when CROMA is *unshipped* — **CROMA is now shipped** | **No** |

Re-running the affected files together passes **137** tests, which is what isolates them as environmental
rather than behavioural (`release/repo/docs/REPRODUCIBILITY.md` §5.4).

**Known per-suite results:**

| Suite | Command | Expected |
|---|---|---|
| Frontend live-wiring | `pytest tests/unit/test_frontend_live_wiring.py` | **106 passed** |
| Doc/frontend suite | `pytest` on the 5 doc/frontend files | **183 passed** |
| Full unit suite | `pytest tests/unit` | 5–6 **environmental** failures |
| Gateway policy | `pytest tests/unit/test_gateway_policy.py` | **51 passed** |
| Evidence engine | `pytest tests/unit/test_evidence_engine.py` | **73 passed** |

(`release/repo/README.md` §The test suites; `docs/PHASE19_FINAL_HARDENING.md` §6.)

> **The precise full-suite *collected* count is `UNKNOWN — not established from the available evidence`.**
> Per-suite counts are known; the single collected total is not (`release/repo/docs/REPRODUCIBILITY.md`
> §5.4).

---

# Part VI — Scripts and notebooks

## 18. The class of scripts under `scripts/`

`scripts/` is a **flat** directory of executable helpers — no sub-packages. The listing holds **51 script
files** (47 `.py`, 3 `.ps1`, 1 `.mjs`) plus `__init__.py`. They fall into clear classes:

| Class | Examples | Purpose |
|---|---|---|
| **Training entry points** | `train_router.py`, `train_grounding.py`, `train_change.py`, `train_fusion.py`, `train_change_vqa.py` | thin CLIs over the `training/` modules (§20) |
| **Evaluation** | `eval_change.py`, `eval_fusion_115.py`, `eval_grounding_head.py`, `evaluate_change_vqa.py` | per-task evaluation |
| **Data preparation** | `prepare_bigearthnet.py`, `prepare_change_vqa.py`, `select_bigearthnet_slice.py` | build corpora and selection manifests |
| **Contract probes** | `probe_grounding_head_contract.py`, `probe_remoteclip_contract.py`, `probe_vlm_contract.py` | measure a real model's contract before relying on it |
| **Smoke tests** | `smoke_test_change.py`, `smoke_test_grounding_head.py`, `smoke_test_vlm.py` | a minimal real path per specialist |
| **Verification / gates** | `verify_gate1.py`, `verify_croma_forward.py`, `verify_grounding_e2e.py`, `verify_levir_real.py`, `verify_cdvqa_imagery.py` | prove a property end to end |
| **Threshold / hyperparameter sweeps** | `sweep_change_threshold.py`, `sweep_router_threshold.py`, `fusion_seed_variance.py` | sweep a frozen knob |
| **Phase-6 (VLM) tooling** | `phase6_train_vlm.py`, `phase6_adjudicate_test.py`, `phase6_close.py`, `phase6_rerule.py`, `phase6_recover_baseline_test.py` | the VLM acceptance workflow |
| **Phase-12 (fusion) tooling** | `p12_integrity_verify.py`, `p12_preflight_verify.py`, `run_phase12_extraction.ps1`, `run_phase12_resume_ref.ps1`, `run_phase12_resume_s1_ref.ps1` | the fusion feature-extraction workflow (PowerShell drivers) |
| **Calibration** | `fit_calibration.py` | fit the temperature-scaling artifact on validation (§22) |
| **Kaggle packaging / rehearsal** | `package_kaggle_code.py`, `rehearse_kaggle_notebook.py`, `rehearse_change_notebook.py` | package and dry-run the notebooks |
| **Deployment validation** | `validate_deploy_config.py` | prove `configs/deploy.yaml` is inert |
| **Frontend staging** | `stage_pages.mjs` | build the Cloudflare Pages bundle |
| **Environment / diagnostics** | `check_env.py`, `diagnose_feature_cache.py`, `croma_normalisation_arm_probe.py`, `analyze_grounding_resolution.py`, `analyze_reben_labels.py`, `check_cdvqa_second_overlap.py`, `establish_cdvqa_temporal_order.py`, `exp_grounding_resolution.py`, `extract_fusion_features.py` | environment and dataset diagnostics |

**The training entry points share one convention**, stated in their docstrings: *"A THIN CLI over
`training/<task>/train.py` … Everything that computes lives in the module; this file parses arguments,
reports the environment, prints the accounting, and returns an exit code."* (`scripts/train_change.py:1-6`,
`scripts/train_fusion.py:1-6`). The exit codes are a contract (`scripts/train_change.py:17-24`):

```
0   training completed (or --dry-run validated a present dataset)
2   the dataset is missing, empty, or cannot be split -- NOT a crash
3   the run started and the trainer raised a typed error
```

> **Re-exported names are part of the contract.** Several scripts re-export their trainer's symbols at
> module scope, and a test asserts it. E.g. `tests/unit/test_change_train_script_contract.py` asserts
> `change_loss`, `train_change_head` and `evaluate` are reachable as `scripts.train_change.*`
> (`scripts/train_change.py:26-30`).

## 19. `notebooks/`

| Notebook | Purpose |
|---|---|
| `kaggle_change_vqa_train.ipynb` | R-02 change-VQA training (§21) |
| `kaggle_phase6_vlm_lora.ipynb` | Phase-6 SmolVLM LoRA training (§21) |
| `kaggle_change_training.ipynb` | the change **detector** — *"frozen and out of scope"* (`docs/R02_KAGGLE_TRAINING_GUIDE.md` §4) |
| `kaggle_grounding_resolution.ipynb` | Phase-7 grounding resolution — unrelated to R-02 |

> **Do not confuse the two change notebooks.** For change-VQA use `kaggle_change_vqa_train.ipynb`; do
> **not** use `kaggle_change_training.ipynb` (that one trains the change *detector*) or
> `kaggle_grounding_resolution.ipynb` (Phase 7) (`docs/R02_KAGGLE_TRAINING_GUIDE.md` §4).

Two `.bak-*` files sit beside the Phase-6 notebook; they are editor backups, not runnable notebooks.

---

# Part VII — Training entry points

The six artifacts train in two places: four locally, two on an external GPU. This is a **deliberate
boundary** — the release ships *"frozen artifacts with provenance, not a retraining harness"*
(`release/repo/docs/REPRODUCIBILITY.md` §8.5).

| Artifact | Where it trains | Reproducible from this release? |
|---|---|---|
| router adapter | local CPU | **yes** — `configs/base.yaml` §`router.training` |
| grounding head | local | **yes** — `configs/base.yaml` §`grounding_training` |
| change head | local | **yes** — `configs/base.yaml` §`change` |
| optical_sar fusion head | local, seed sweep | **yes** |
| change_vqa head | **external GPU (Kaggle)** | **partly** — the promotion gate, evaluation and serving wiring are reproducible; there is no one-command retrain |
| vlm LoRA adapter | **external GPU** | **partly** — same |

(`release/repo/README.md` §What "reproduce" means.)

## 20. Local training (router, grounding, change, optical-SAR)

### 20.1 Router — CPU-only, and fast

```bash
python scripts/train_router.py --smoke           # fast, stub encoder
python scripts/train_router.py --stub            # fast, no model download
python scripts/train_router.py                   # full run, real MiniLM
```

(`scripts/train_router.py:1-10`)

The script's own measured note: *"Runs entirely on CPU. Measured cost with the real encoder: ~40 s to load
MiniLM on a cold cache, ~0.1 s to embed the corpus, ~1 s to train 60 epochs. There is no reason to spend
Kaggle GPU quota on this."* The encoder is frozen, so embeddings are cached and the adapter trains on
cached vectors (`configs/base.yaml:67-69`).

### 20.2 Grounding — two separable stages

```bash
# stage 1 only — measure the corpus before committing to training
python scripts/train_grounding.py --data-root <root> --extract-only

# both stages, real run
python scripts/train_grounding.py --data-root <root> --checkpoint <ckpt.pt>

# 2-batch forward+backward+checkpoint+reload, on CPU
python scripts/train_grounding.py --data-root <root> --debug
```

(`scripts/train_grounding.py:6-22`)

Splitting extraction from training means a hyperparameter change does not re-encode 15,699 images, and a
crashed training run does not lose the cache. The stated bar: *"the Phase 7 zero-shot baseline scored
0.0972 mean best IoU over 16,159 real eval records. This head must beat it by MIN_IMPROVEMENT_IOU to
justify existing. The comparison is printed whether or not it passes."*

### 20.3 Change — a thin CLI over `training/change/train.py`

```bash
# validate the data and the split; build nothing, train nothing
python scripts/train_change.py --data-root <LEVIR-CD root> --dry-run

# real run on CPU
python scripts/train_change.py --data-root <LEVIR-CD root> --device cpu

# a first real-data run that does not commit an hour
python scripts/train_change.py --data-root <root> --limit 256 --epochs 3
```

(`scripts/train_change.py:8-16`)

`--dry-run` is *"the leakage check without the cost: it loads the items, performs the scene-disjoint
split, runs `assert_image_disjoint`, prints the accounting and exits. It does NOT build the detector, so
it cannot trigger a pretrained-weights download, and it writes no files."*

### 20.4 Optical-SAR fusion — CPU-only, no result claimed

```bash
# validate the caches and the split; train nothing, write nothing
python scripts/train_fusion.py --train-cache train.npz --val-cache val.npz --dry-run

# a real run on CPU (the fusion head is CPU-only)
python scripts/train_fusion.py --train-cache train.npz --val-cache val.npz --arm A --epochs 20
```

(`scripts/train_fusion.py:8-15`)

> **No performance number is a result here.** The script states it: *"This loop has never seen the real
> paired BigEarthNet-S1+S2 corpus. Every run record it writes carries `result_status` and
> `pre_registered_metric_computed: false`; the pre-registered 11.5 metric is not computed."*
> (`scripts/train_fusion.py:17-20`.)

## 21. External GPU training (change-VQA, VLM LoRA)

Both external runs happen on **Kaggle** with **GPU T4 ×2**. Neither is a one-command retrain from this
release.

### 21.1 Change-VQA (R-02)

**Guide:** `docs/R02_KAGGLE_TRAINING_GUIDE.md` (37 KB). **Runbook:** `RUNBOOK_CHANGE_VQA_KAGGLE.md`.

| Property | Value | Source |
|---|---|---|
| notebook | `notebooks/kaggle_change_vqa_train.ipynb` | `docs/R02_KAGGLE_TRAINING_GUIDE.md` §4 |
| accelerator | **GPU T4 ×2** | §5 |
| internet | **On** (MiniLM downloads) | §5 |
| cells | **all 34, top to bottom** | §7 |
| hard stop | `HARD_STOP_SECONDS = 3 * 3600` (the plan's 3 h) | §7 |
| trainer defaults | `epochs=40, batch_size=128, seed=42, patience=6, time_limit=10800s` | §7 |

The path-discovery cells find the code root (four markers) and the CDVQA root (12 annotations + 4 image
dirs), and section 3c verifies the frozen STANet checkpoint by size and SHA256
(`RUNBOOK_CHANGE_VQA_KAGGLE.md` §5). **Test splits are not readable by the trainer:**
`training/change_vqa/train.py` loads only Train and Val, and *"there is no option in either file that
changes that"* (`scripts/train_change_vqa.py:11-15`).

### 21.2 VLM LoRA (Phase 6)

**Runbook:** `RUNBOOK_PHASE6_VLM_KAGGLE.md`.

| Property | Value | Source |
|---|---|---|
| notebook | `notebooks/kaggle_phase6_vlm_lora.ipynb` | `RUNBOOK_PHASE6_VLM_KAGGLE.md` §1 |
| accelerator | **GPU T4 ×2** | §4 |
| internet | **off** (base model attached as input) | §4 |
| precision | **`fp16`, not the plan's `bf16`** — T4 is SM 7.5 | §4 |
| base model | `HuggingFaceTB/SmolVLM-500M-Instruct`, ~1.02 GB `model.safetensors` | §4 |

The trainer raises a `ConfigError` on `bf16` rather than silently falling back, and the deviation is
recorded in the manifest under `plan_deviations` (`RUNBOOK_PHASE6_VLM_KAGGLE.md` §4).

> **The VLM adapter's status is `ACCEPTANCE-REJECTED`.** Its metrics are *usable* (`exact_match 0.963`)
> but it was not promoted. **USABLE ≠ ACCEPTED** (`DOCS_STYLE_GUIDE.md` §3). Do not describe the VLM path
> as accepted.

## 22. Calibration

```bash
python scripts/fit_calibration.py --dry-run    # check inputs, exit
python scripts/fit_calibration.py              # fit, write artifact
```

(`scripts/fit_calibration.py:19-21`)

It fits on **validation** data, and both the fitter entry point and the artifact writer **refuse a
held-out split** — *"Fitting on Test or Test2 would make the reported confidence a function of the answers
it is used to score — a leak, not a calibration."* (`scripts/fit_calibration.py:7-13`.)

> **Calibration made ECE worse** — `0.013755 → 0.014929` — and is **retained only because it is in the
> frozen config**. Never present it as an improvement (`DOCS_STYLE_GUIDE.md` §3).

---

# Part VIII — Coding conventions

## 23. What the code actually does

These conventions are visible across the files read. They are not aspirational style rules; each is
observable in the code.

### 23.1 Every module has a substantial docstring stating *why*

The files read are densely commented at the module and function level, and the comments explain decisions
and failure modes rather than restating the code. Examples: `core/registry.py:1-76` (a 76-line module
docstring on the capability-key trap), `deploy/codespace/launch.sh:1-24` (why the launcher is defensive),
`gateway/app.py:53-106` (the annotation-scope trap). **A change that removes the reasoning from a
comment removes the reason a future reader will not re-introduce the bug.**

### 23.2 `from __future__ import annotations` is standard

It appears at the top of nearly every module (`core/config.py:11`, `core/registry.py:78`,
`specialists/base.py:19`, `gateway/app.py:48`, `deploy/render/main.py:40`, `deploy/render/codespaces.py:16`,
`deploy/codespace/warm_cache.py:28`, `scripts/*.py`). It is convenient — and it is the direct cause of
Trap 6 (§30).

### 23.3 Findings are recorded as short codes, inline

The code refers to findings by code (`C-1`, `C-6`, `C-8`, `F5-2`, `P7-1`, `F-15`, `F-16`, `F-17`) and
states the failure each guard prevents. This is a documentation convention enforced by comments, e.g.
`core/config.py:121-142` (finding F5-2, the 17× overrun), `core/registry.py:227-235` (F-17, a config
surface removed rather than left as a silent no-op).

### 23.4 Pydantic models set `extra="forbid"` and carry validators

Every schema model forbids extra fields and uses `@field_validator` / `@model_validator` for cross-field
rules (§11). New models should follow the same pattern.

### 23.5 Loggers are module-level and named `satquery.<area>`

`logging.getLogger("satquery.orchestrator")` (`deploy/render/main.py:74`),
`logging.getLogger("satquery.registry")` (`core/registry.py:95`), `logging.getLogger(__name__)`
(`gateway/app.py:123`). Log calls use `%s`-style free text — there are no structured/JSON logs
(`docs/architecture/10-observability-and-ops.md` §4.4).

### 23.6 Client-visible strings are scrubbed

Server-side diagnostics that name absolute paths must not reach a client. `core/errors.scrub_paths` reduces
a path to a basename before it is published (`core/registry.py:297-312,560-598`;
`core/errors.py:39`). Any new client-visible field that could carry a path or an exception string must go
through the same scrub.

### 23.7 `noqa` comments state the reason

Where a lint suppression is used, the reason is written, e.g. `# noqa: E402  (see comment above)` in
`gateway/app.py:85-106`, and `# noqa: BLE001 - report, never crash the warm step` in
`deploy/codespace/warm_cache.py:74`.

### 23.8 Tests assert the *cause*, not the symptom

The regression test for the annotation-scope trap asserts that the route has no query parameters and that
`gateway.app.Request is starlette.requests.Request` — *"rather than the symptom, because asserting the
symptom would be brittle"* (`docs/STEP7_BACKEND_CHAIN_REPORT.md` §13). Follow this when writing a
regression test.

### 23.9 The 88-column soft limit

The files read wrap around 88 characters. There is no committed linter config in the files read, so this
is a convention, not an enforced rule — **the exact formatter/linter configuration is `UNKNOWN — not
established from the available evidence`.**

---

# Part IX — Known development traps

## 24. The traps, in one table

| # | Trap | One-line consequence |
|---|---|---|
| 1 | the monorepo `deploy/` is stale and untracked | it is **not** the deployed source |
| 2 | the sandbox proxy is dead | outbound calls need `--noproxy '*'` / `ProxyHandler({})` |
| 3 | pytest exists only in the venv | the system `pytest` resolves to the wrong interpreter |
| 4 | Chrome drops synthetic CDP key events without OS focus | a browser-driven run silently answers the default query |
| 5 | Cloudflare `_headers` rules concatenate | a later rule cannot "fix" an earlier one |
| 6 | `from __future__ import annotations` + FastAPI | an unresolvable `Request` annotation becomes a **required query parameter** |
| 7 | the full-suite run trips the bulk-delete guard | 4 spurious `test_safe_delete_shim` failures |
| 8 | a stale serve process keeps answering | it reports the **previous revision's** capabilities |

## 25. Trap 1 — the stale untracked `deploy/`

**Symptom.** You edit `deploy/render/main.py`, deploy your change, and nothing changes in production — or
you read `deploy/render/main.py` and cannot find the tunnel code the live service runs.

**Root cause.** `deploy/` inside the monorepo is **stale and untracked**. `git status` reports
`?? deploy/` (verified in the working copy). It is **not** the deployed source
(`release/repo/docs/DEPLOYMENT.md` §1; `release/CURRENT_RELEASE_STATE.md` §6).

**Evidence of divergence.** The monorepo `deploy/render/main.py` (532 lines) exposes `/api/health` with a
`config` block that has **no** `tunnel` field and no `transport_mode` / `tunnel_timeout_s` /
`wake_timeout_s` keys (`deploy/render/main.py:444-466`), whereas the **live** payload carries all of them
(`release/repo/docs/DEPLOYMENT.md` §5). The monorepo copy also lacks `tunnel_agent.py` and `doctor.sh`,
both of which `launch.sh` references (`deploy/codespace/launch.sh:83,89,153,160-168,179`).

**Fix.** Fetch the deployed file from the private repository and diff before editing. Treat the monorepo
`deploy/` as documentation of intent, not as source.

## 26. Trap 2 — the dead sandbox proxy needs `--noproxy '*'`

**Symptom.** Outbound HTTP calls fail or hang in the authoring sandbox.

**Root cause.** The sandbox proxy is dead; requests are routed to it and never reach the target.

**Fix.** Disable proxies for the call (`release/repo/docs/REPRODUCIBILITY.md` §10.1):

```bash
curl --noproxy '*' https://<backend-host>/api/health
```

```python
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
```

`release/tools/hf_verify.py:37-38` does exactly this (`opener_no_proxy()`). In a normal environment the
flag is harmless; in the sandbox it is mandatory.

## 27. Trap 3 — pytest only in the venv

**Symptom.** Invoking the system `pytest` fails or resolves to a different interpreter.

**Root cause.** pytest is installed only in `.venv`.

**Fix.** Always invoke the venv interpreter explicitly (`release/repo/docs/REPRODUCIBILITY.md` §10.2):

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q
```

## 28. Trap 4 — Chrome drops synthetic CDP key events

**Symptom.** A browser-driven run silently answers the *default* query; the query box looks untouched;
`mock_nodes` is non-zero.

**Root cause.** Chrome **drops synthesized key events when the browser window does not hold OS focus**.
`press_key` / `fill_input` (real CDP key events) are focus-gated; `Input.insertText` (`type_text`) is not.

**Measured.** With Chrome backgrounded, `press_key("Z")` left `#qtext.value` unchanged, while
`type_text("Q")` inserted fine (`release/repo/docs/REPRODUCIBILITY.md` §6.4).

**Fix.** Use `type_text` (not `fill_input`), and **assert the input state before dispatch** — `q_ok`
(the query box really held the query), `obs_ok` (`#obsTail == 'ready'`), `t0_ok` (both frames present
where required). This is the single most dangerous trap because it produces a **silent false pass** — the
pipeline "works", it just answered a different question
(`release/repo/docs/REPRODUCIBILITY.md` §10.5).

## 29. Trap 5 — Cloudflare `_headers` concatenate

**Symptom.** A specific cache-control rule does not take effect; the browser caches a file you expected it
to revalidate.

**Root cause.** Cloudflare `_headers` rules **concatenate, they do not override.** Two matching rules are
merged: a specific rule nested under a broad `/assets/img/*` rule yields
`max-age=604800, …, max-age=0, must-revalidate` — and **Chromium takes the FIRST `max-age`**. The file's
own "later rules override" comment is **false** (`release/repo/docs/DEPLOYMENT.md` §10).

**Fix.** Order rules so the *broadest* rule appears last, and never rely on a later rule overriding an
earlier one. Related: Cloudflare **308-redirects `X.html` → `/X`**, so reference the extensionless path
(`release/repo/docs/REPRODUCIBILITY.md` §10.4).

## 30. Trap 6 — the annotation-scope trap

**Symptom.** Every `POST` route returns `422` with FastAPI's own shape, without ever entering the handler:

```
POST /v1/analyze  →  422
{"detail":[{"type":"missing","loc":["query","request"],"msg":"Field required"}]}
```

(`docs/STEP7_BACKEND_CHAIN_REPORT.md` §13.)

**Root cause.** The module uses `from __future__ import annotations`, so every annotation is a **string**
at runtime. FastAPI resolves those strings via `get_typed_signature`, which calls
`eval(annotation, func.__globals__)`. If `Request` is imported **inside** `create_app`, the route closures
capture the name as a *local* of `create_app`; it never appears in `gateway.app.__globals__`, so
resolution fails and FastAPI is left holding a bare `ForwardRef('Request')`. **FastAPI does not raise** —
it silently falls back to treating the parameter as a **required query parameter named `request`**
(`gateway/app.py:53-73`).

Three things were wrong at once: the request body was never read, the gateway's own validation never ran,
and the error shape was FastAPI's `{"detail": ...}` rather than the contract's `{"error": {...}}`. **Every**
POST route was affected, including `/v1/assets`.

**The asymmetry to internalise.** An unresolvable **parameter** annotation is *silently reinterpreted*
(the handler never runs), while an unresolvable **return** annotation *raises*
(`pydantic.errors.PydanticUndefinedAnnotation: name 'JSONResponse' is not defined`, which made
`create_app()` unbuildable). Same root cause, opposite diagnosability
(`gateway/app.py:88-106`).

**Fix.** Bind the annotation subjects at **module scope** (`gateway/app.py:79-106`):

```python
if TYPE_CHECKING:  # pragma: no cover
    from starlette.requests import Request
    from starlette.responses import Response

#: Runtime bindings used as annotation subjects in this module. Deliberately
#: module-level so `eval()` can find them. See the comment above.
from starlette.requests import Request  # noqa: E402  (see comment above)
from starlette.responses import Response  # noqa: E402  (see comment above)
from fastapi.responses import JSONResponse  # noqa: E402  (see comment above)
```

The regression test asserts the **cause**: the route has no query parameters, and
`gateway.app.Request is starlette.requests.Request`.

> **The general rule.** Any name used as a FastAPI route annotation in a module with
> `from __future__ import annotations` **must** be importable from that module's globals. Import it at
> module scope, not inside the factory.

## 31. Trap 7 — the full-suite bulk-delete guard

**Symptom.** Running the whole `tests/unit` tree trips 4× `test_safe_delete_shim` failures.

**Root cause.** Windows **verbatim-path** defects in the sandbox's bulk-delete guard. The precise
condition under which the shim intermittently triggers on Windows is `UNKNOWN — not established from the
available evidence` (`release/repo/docs/REPRODUCIBILITY.md` §10.3).

**Fix / workaround.** Run the targeted suites (106 and 183 pass cleanly); treat the 4 shim failures as
environmental, not regressions (§17).

## 32. Trap 8 — the stale serve process

**Symptom.** The Codespace answers `/v1/health` and `/v1/capabilities`, but reports the **previous
revision's** capabilities.

**Root cause.** A serve process started before a code or environment change keeps serving from old code.
The serve process reads its environment exactly once, at startup (`deploy/codespace/launch.sh:100-102`).

**Fix.** `launch.sh` already handles it: it records a **stamp** of the revision and the asset
configuration and restarts the server when the stamp disagrees
(`deploy/codespace/launch.sh:100-147`). The rule for a developer: **a restart, not a reload, is required
after any env or revision change.**

> *"A stale serve process is worse than no process: it answers /v1/health and /v1/capabilities from OLD
> code, so the deployment looks alive while reporting the previous revision's capabilities."*
> (`deploy/codespace/launch.sh:111-113`)

### Related platform traps worth knowing

| Trap | Detail |
|---|---|
| a forwarded Codespace port returns `302` for a private repo | this is *why* the outbound tunnel exists (`release/repo/docs/DEPLOYMENT.md` §10) |
| the tunnel agent must be started by the devcontainer `postStartCommand` | a restarted Codespace comes up with `agent_connected: false` otherwise |
| never retry `POST /api/infer` at the gateway | a retry consumes inference twice |
| `containerEnv` applies only at container **creation** | an env change needs a restart, which is why `launch.sh` re-exports on every start (`deploy/codespace/launch.sh:49-52`) |
| the Codespace filesystem is **ephemeral** | uploaded assets and logs vanish with the Codespace (`deploy/codespace/launch.sh:44-47`) |

---

# Part X — Status and evidence

## 33. `NOT RUN` / `OPEN` / `BLOCKED` / `UNKNOWN` for development

| # | Item | Status |
|---|---|---|
| 1 | B-07 — tunnel gaps; patch prepared, **not deployed** | **`OPEN`** |
| 2 | B-02 — `/api/health` `codespace_name` trailing `\n` | **`OPEN` (cosmetic)** |
| 3 | A `LICENSE` file | **`OPEN`** — **no LICENSE file exists**; README says to add one before public release |
| 4 | The monorepo README | **materially stale** — it calls the frontend "hermetic", describes a 4-endpoint `/v1/*` contract, omits the tunnel, and points at the stale `deploy/` (`release/CURRENT_RELEASE_STATE.md` §6) |
| 5 | `hf/SETUP.md` and `hf/README.md` | **stale** — they assert the project ships no weights, which is now false (`release/CURRENT_RELEASE_STATE.md` §6) |
| 6 | The exact formatter / linter configuration | **`UNKNOWN`** — no committed config in the files read (§23.9) |
| 7 | The precise full-suite collected test count | **`UNKNOWN`** — per-suite counts are known, the total is not |
| 8 | The exact Windows trigger for the delete-shim flake | **`UNKNOWN`** |
| 9 | Whether `doctor.sh` / `tunnel_agent.py` exist in the deployed inference repo | **`UNKNOWN`** — the monorepo copy lacks them |
| 10 | A system-level end-to-end benchmark | **`NOT RUN`** — none exists |
| 11 | The router **test**-split number | **`NOT RUN`** — only validation (`n = 86`, ungated) exists |
| 12 | Captioning benchmark | **`NOT RUN`** — implemented, not benchmarked |
| 13 | A one-command retrain for the two external artifacts | **not implemented** — deliberate boundary (§21) |
| 14 | `torch.compile`, CUDA, quantisation, thread capping | **not implemented** — CPU-first by design (§1) |
| 15 | Enforcement of a one-model cache (`cache_max_models: 1`) | **UNVERIFIED** — the value is reported, not proven enforced (`docs/DEPLOYMENT_DECISION.md` §5) |

> **The stale-README trap is worth its own line.** `README.md` in the monorepo is *"materially stale"*: it
> calls the frontend *"hermetic — no backend calls"* (it calls `/api/*` on Render), puts Render/Codespace
> as *"in progress"* (both deployed), describes a 4-endpoint `/v1/*` contract (the live contract is
> `/api/*`), omits the tunnel, and points at the stale untracked `deploy/` as the deployment source
> (`release/CURRENT_RELEASE_STATE.md` §6). The public release documentation is authoritative; the
> monorepo README is not.

## 34. Where the evidence lives

| What | Where |
|---|---|
| the frozen config, invariants and hash | `configs/base.yaml`; `core/config.py:76-80,94-222` |
| the binding schemas | `core/schemas.py` |
| the specialist interface | `specialists/base.py` |
| the capability table and lazy construction | `core/registry.py` |
| the planner mapping | `core/planner.py:121-142` |
| the dependency profiles and contract notes | `requirements.txt` |
| the test markers | `pytest.ini` |
| the launcher and warm-up | `deploy/codespace/launch.sh`, `deploy/codespace/warm_cache.py` |
| the annotation-scope trap | `gateway/app.py:53-106`; `docs/STEP7_BACKEND_CHAIN_REPORT.md` §13 |
| the deploy manifest is inert | `docs/PHASE18_DEPLOYMENT_PACKAGING.md`; `scripts/validate_deploy_config.py` |
| the change-VQA Kaggle guide | `docs/R02_KAGGLE_TRAINING_GUIDE.md`; `RUNBOOK_CHANGE_VQA_KAGGLE.md` |
| the VLM LoRA Kaggle runbook | `RUNBOOK_PHASE6_VLM_KAGGLE.md` |
| the test suites and their expected results | `release/repo/docs/REPRODUCIBILITY.md` §5 |
| the environment traps | `release/repo/docs/REPRODUCIBILITY.md` §10 |
| the platform traps | `release/repo/docs/DEPLOYMENT.md` §10 |
| the live topology, env vars, cold start | `release/repo/docs/DEPLOYMENT.md` |
| the operations manual | [../OPERATIONS.md](OPERATIONS.md) |
| the factual inventory | `release/CURRENT_RELEASE_STATE.md` |

### Cross-references

| For | See |
|---|---|
| the frozen config and `Config.hash == 78f1e3700da15aa1` | [07 — Configuration and Freeze](architecture/07-configuration-freeze.md) |
| the specialist contract in depth | [05 — Specialists](architecture/05-specialists.md) |
| the router's five heads and the label space | [04 — Router](architecture/04-router.md) |
| the evidence and confidence stages | [06 — Evidence and Confidence](architecture/06-evidence-and-confidence.md) |
| the request lifecycle and the nine-state spine | [03 — Request Lifecycle](architecture/03-request-lifecycle.md) |
| the four endpoints and error codes | [08 — The API Contract](architecture/08-api-contract.md) |
| how to operate the live stack | [../OPERATIONS.md](OPERATIONS.md) |
| per-artifact hyperparameters | [../TRAINING.md](TRAINING.md) |
| what a third party can and cannot reproduce | [../REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

---

> **Chapter summary.** Python 3.11+ and a CPU are sufficient; there is no CUDA requirement, and all
> placement is `.to(device)`, never `.cuda()`. The single registry is `configs/base.yaml`, its hash
> **`78f1e3700da15aa1`** is frozen, and **editing the config moves the hash and invalidates every artifact
> keyed to it** — the one non-reversible action in the repository. `core/schemas.py` is the binding
> contract and every specialist returns exactly `SpecialistResult`; the specialist interface is four
> methods in `specialists/base.py`, and a new specialist is added by implementing it and adding a
> `SpecialistSpec` row keyed on **capability**, not name. Tests run from the venv, targeted, because the
> full suite trips the sandbox's bulk-delete guard. Four artifacts train locally; two — change-VQA and the
> VLM LoRA — train on an external GPU and are not one-command reproducible. Eight development traps are
> recorded, of which the most dangerous are the stale untracked `deploy/`, the annotation-scope trap that
> turns a `Request` parameter into a required query parameter, and the Chrome focus trap that produces a
> silent false pass. `NOT RUN`/`OPEN` items include B-07 and B-02 (both `OPEN`), the absent `LICENSE`, the
> stale monorepo README and `hf/` docs, and several `UNKNOWN — not established from the available
> evidence` gaps.
