# SatQuery AI — Serving

**Chapter scope.** This chapter documents the SatQuery AI inference service end to end: how the
service is composed, the four-endpoint contract it exposes and the `/api/*` mirror of that contract,
the entrypoint requirements any host must satisfy, the lazy model-loading model, the annotation-scope
defect that once made every upload fail with a `422`, the ephemeral asset store, the error taxonomy
and its machine codes, the tunnel transport, and — stated plainly — what the service does *not* do.

**Grounding.** Every claim below comes from a file that was read for this chapter, cited inline, e.g.
`(app/space_app.py)`, `(docs/API_CONTRACT.md §2.4)`. No endpoint, field, environment variable, status
code, or number is invented. Where the evidence does not exist, the text says exactly:
`UNKNOWN — not established from the available evidence`.

**Status vocabulary** follows `release/DOCS_STYLE_GUIDE.md` §2: `IMPLEMENTED` · `VERIFIED` ·
`MEASURED` · `ATTEMPTED` · `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` ·
`CLOSED`.

**Nothing in this chapter is a system-level accuracy claim.** Per `release/DOCS_STYLE_GUIDE.md` §3
there is **no end-to-end benchmark** for SatQuery AI. This chapter describes a *service*; it does not
score one.

---

## 1. What the serving tier is

SatQuery AI's serving tier is a **Python HTTP service** that exposes the project's analysis capability
over four endpoints. It is built on FastAPI/Starlette, it is served by `uvicorn`, and it is composed
by three modules:

| Module | Role |
|---|---|
| `app/serving.py` | The **composition root**: builds a deployable registry and controller, wiring trained artifacts through the registry's `builders=` seam. |
| `app/space_app.py` | The **HTTP application**: builds the FastAPI app (`build_space_app()`), owns the four routes, the asset store, and the error handlers. |
| `app/deployment.py` | The **capability adapter**: turns internal registry state into the public capability vocabulary and produces the health and capabilities payloads. |

Around those three sit:

- `core/controller.py` — `AnalysisController`, the control tier that runs the pipeline.
- `core/registry.py` — `SpecialistRegistry`, which discovers specialists from a spec table and builds
  them lazily.
- `core/planner.py` — `PolicyPlanner`, the deterministic policy planner.
- `core/errors.py` — the error taxonomy (23 codes) and the path scrubber.
- `gateway/app.py` + `gateway/policy.py` — the gateway (an optional front tier; see §3.3).
- `deploy/codespace/serve.py` — the 26-line process entrypoint that calls `build_space_app()`.
- `deploy/codespace/launch.sh` — the launcher that starts the service and its tunnel agent.
- `deploy/render/main.py` — the Render orchestrator that exposes the `/api/*` mirror.

The service's job is narrow and worth stating: **accept an analysis request, run the pipeline, return a
`ResultEnvelope`.** It does not render a UI, it does not stream, and it does not persist results. §11
lists what it does not do in full.

---

## 2. The composition root: `app/serving.py`

`app/serving.py` is 275 lines. Its module docstring calls itself "the public serving entry point" and
states that it is "the thin, public composition root that wires a *deployable* controller".

### 2.1 The three wired artifacts

The module declares three module-level `Path` constants. Each is a *repo-local artifact identity*, not
a config key:

| Constant | Path (relative to `REPO_ROOT`) |
|---|---|
| `CHANGE_CHECKPOINT` | `artifacts/change/levir_change_v001/head.pt` |
| `CHANGE_VQA_HEAD` | `artifacts/change_vqa/run/head.pt` |
| `FUSION_HEAD` | `artifacts/optical_sar/fusion_head_production_v001/head.pt` |

Their documented identities, as stated in the module's comments:

- **`CHANGE_CHECKPOINT`** — "The trained, benchmarked change head (test pooled IoU 0.8122)." It is
  "the single source of truth for where serving looks for it; tests monkeypatch this to simulate an
  absent artifact." It is also the detector whose features `scripts/prepare_change_vqa.py` builds, "so
  training and serving share it." A test
  (`tests/unit/test_app_serving.py::test_serving_and_preparation_share_one_stanet_checkpoint`) keeps
  the two literals equal and fails if either side drifts.
- **`CHANGE_VQA_HEAD`** — "The R-02 change-VQA reasoning head." Written by
  `scripts/train_change_vqa.py` (`--output-dir`, default `artifacts/change_vqa/run`) and read by
  `scripts/evaluate_change_vqa.py` (`DEFAULT_CHECKPOINT`, the same path). It "does NOT exist in a fresh
  checkout: it is produced by the external Kaggle run and returned to the maintainer". Absent ⇒ the
  specialist constructs, reports itself unavailable, and answers nothing.
- **`FUSION_HEAD`** — "The verified production optical-SAR fusion head (Phase 12). Its identity is
  pre-registered, not inferred: sha256
  `785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab`, 14,427,457 bytes, 1,201,711
  parameters, and the checkpoint self-identifies with the embedded `config_hash`
  `78f1e3700da15aa1` and `arm='A'`."

Note the last one carefully: the fusion head's *self-identification* carries the same frozen config
hash `78f1e3700da15aa1` that `release/DOCS_STYLE_GUIDE.md` §3 records as the project's frozen config
hash. The artifact and the config agree by construction.

### 2.2 Why artifacts are wired through `builders=`, not through config

This is the single most important design decision in the serving tier, and `app/serving.py` documents
it at length. The mechanism:

`Config.hash` (`core/config.py:79-80`) is a **sha256 over the WHOLE registry**. Adding one key moves
the hash. The shipped change head records `78f1e3700da15aa1` in
`artifacts/change/levir_change_v001/model_metadata.json`, and `scripts/eval_change.py` **refuses to
score on a hash drift (exit 3)**.

Therefore: editing `configs/base.yaml` to point at a trained head would **invalidate the project's own
benchmark number**. The supported wiring path is instead the registry's `builders=` override
(`core/registry.py:420-433`), keyed by spec name — `"change"` (`core/registry.py:204-213`) — and it is
a **call-site argument, not config**, so the hash is untouched.

The registry calls a builder as `builder(self.config, **kwargs)` and only passes config keys that
resolve (`_builder_kwargs`, `core/registry.py:435-453`). Since `change.checkpoint_path` is unset,
nothing arrives and the real builder would degrade; the override supplies the missing argument.

Stated as a general rule: **in this project, a serving-side artifact path may not be added to
`configs/base.yaml`, because doing so would move the frozen config hash and invalidate every benchmark
number keyed to it.** The `builders=` seam is the hash-exempt channel for such paths.

### 2.3 Degrade, do not crash

The module's docstring states the governing principle: "A serving path must run even when the artifact
is absent."

The distinction it enforces is precise:

- **Absent** is a *deployment case*. When the checkpoint does not exist the module applies NO override,
  so the registry resolves the real builder with no `checkpoint_path` and the documented `DEGRADED`
  contract applies (`specialists/change/specialist.py`).
- **Corrupt** is a *defect*. The real builder still surfaces it as `ModelLoadError` — "the two are
  deliberately not conflated."

The `change_vqa` override is applied **unconditionally**, because its builder's contract is
finer-grained: a missing head and a missing detector are both *named* refusals
(`ChangeVQASpecialist.has_head`, `unavailable_reason`), so wiring it can never turn "absent" into a
crash. Both paths are passed as `None` when the file does not exist, "which the builder reads as
'artifact genuinely absent' rather than 'path I was told about is broken'."

### 2.4 The three builders

**`_wired_change_builder(config, **kwargs)`** — imports `build_change_specialist` lazily ("so importing
`app.serving` stays cheap and does not pull the model stack (torch) into a process that never serves a
change query"), sets `kwargs["checkpoint_path"] = str(CHANGE_CHECKPOINT)`, and delegates.

**`_wired_change_vqa_builder(config, **kwargs)`** — exists to close a **train/serve skew** finding
(named "finding F2" in the code). The skew: `scripts/prepare_change_vqa.py` builds its change features
from the *trained* STANet (`DEFAULT_CHANGE_CHECKPOINT`), but serving had no equivalent wiring —
`change.checkpoint_path` is unset in `configs/base.yaml`, so the registry passed no `checkpoint_path`
and the specialist "would construct an UNTRAINED STANet and answer from a representation the head was
never fitted on." The class's `feature_spec_mismatch()` already refused to answer on that skew, "so
the failure was loud rather than silent — but a deployment that can never answer is still not a
deployment." The override supplies the SAME checkpoint `_wired_change_builder` uses, so the detector
backing a change answer and the detector behind the head's training features are "one artifact by
construction; the spec check stays armed as the second line of defence, not the only one."

**`_wired_optical_sar_builder(config, **kwargs)`** — exists to close a different structural defect:
"the encoder was unreachable by default." The mechanism, quoted from the module:
`specialists/optical_sar/specialist.py:946` builds CROMA only when handed a `checkpoint_path` that
exists:

```python
if checkpoint_path is not None and Path(checkpoint_path).exists():
```

`croma.checkpoint_path` is not in `configs/base.yaml` — and must not be, or `Config.hash` moves — so
the registry passed no `checkpoint_path`, the gate was `False`, and the default serving composition
ran with `encoder=None`. The registry then correctly reported `DEGRADED` ("no encoder; running on
fallback"): "a deployment that could never answer an optical/SAR question. The checkpoint was present
on disk the whole time; nothing asked for it."

The fix resolves the checkpoint from the PINNED identity through the hash-exempt channel
(`croma.resolve_checkpoint_path`: **env → config → pinned Hub cache, offline first**), then hands it to
the real builder. The resolution is recorded as `source` and **logged**, not attached to the returned
specialist — with an explicit reason given in the code: "An unread attribute on a production object is
how a contract quietly grows a second, undocumented shape; and it must not be published either,
because the trace reaches the client and v1 has no auth."

That last sentence is a design principle worth extracting: **anything the trace carries is public,
because v1 has no auth.** So composition-time facts that must not leak are logged rather than attached.

The fusion head is wired the same way for the same reason: `has_head`
(`specialists/optical_sar/specialist.py:175-187`) is `False` without it, so the capability would stay
`DEGRADED` even with the encoder loaded. Absent head ⇒ `None` ⇒ degrade.

### 2.5 `build_serving_registry()`

```python
def build_serving_registry(config=None, *, device=None) -> SpecialistRegistry
```

- `config` — the central `core.config.Config`. "Loaded unchanged when omitted; never mutated."
- `device` — a torch device string; defaults to `config.device_preference`.
- Returns "a `SpecialistRegistry` that constructs nothing yet (`discover()` reads a spec table)."

Its body builds a `builders` dict:

```python
builders = {
    "change_vqa": _wired_change_vqa_builder,
    "optical_sar": _wired_optical_sar_builder,
}
if CHANGE_CHECKPOINT.exists():
    builders["change"] = _wired_change_builder
return SpecialistRegistry.discover(cfg, device=device, builders=builders)
```

Note the asymmetry and the reason for it, which the code comments on: `change_vqa` and `optical_sar`
are registered **UNCONDITIONALLY**, unlike `change`. The comment explains: "The resolver decides at
build time whether an artifact exists, so gating registration on a path this module does not know yet
would be circular. Wiring it cannot turn 'absent' into a crash: the resolver returns `None`, the
builder degrades, and a construction failure is retained as an `UNAVAILABLE` entry by
`SpecialistRegistry.build` rather than escaping."

So: **`change` is gated on the checkpoint existing; `change_vqa` and `optical_sar` are not, because
their builders accept `None` as "absent".**

### 2.6 `build_serving_controller()`

```python
def build_serving_controller(config=None, *, device=None) -> AnalysisController
```

It is "Constructed with `registry=`, `planner=` and `config=` only."

```python
registry = build_serving_registry(cfg, device=device)
return AnalysisController(
    registry=registry,
    planner=PolicyPlanner(registry),
    config=cfg,
)
```

The critical documented consequence: **"No router is attached, so a caller drives it with
`AnalysisRequest(..., force_task=...)`; a natural-language router can be supplied by the caller's own
composition if the router weights are available."**

This is the single most important behavioural fact about the serving tier's request handling: **the
deployed service is driven by an explicit `force_task`, not by natural-language routing.** It explains
why the frontend's `interpret()` (see the `FRONTEND.md` chapter) does the lexical routing in the
browser and then sends a `force_task`: the browser-side interpretation is what fills the gap left by
the deliberately router-less serving composition.

`__all__` exports `CHANGE_CHECKPOINT`, `CHANGE_VQA_HEAD`, `FUSION_HEAD`, `build_serving_controller`,
and `build_serving_registry`.

---

## 3. The HTTP application: `app/space_app.py`

`app/space_app.py` is 736 lines and owns the HTTP surface.

### 3.1 The four routes

`build_space_app()` assembles a FastAPI application with four routes:

| Method | Path | Kind | Notes |
|---|---|---|---|
| `GET` | `/v1/health` | cheap | Health block; includes `device` and `gpu_available`. |
| `GET` | `/v1/capabilities` | cheap | Capability block; per-task availability and reasons. |
| `POST` | `/v1/analyze` | **COSTLY** | Runs the pipeline; returns a `ResultEnvelope`. |
| `POST` | `/v1/assets` | **COSTLY** | Uploads an asset; returns an opaque `asset_id`. |

The "cheap vs COSTLY" distinction is not decoration: `gateway/app.py` declares

```python
COSTLY_ROUTES = ("/v1/analyze", "/v1/assets")
```

and the gateway's policy (`gateway/policy.py`) applies its body-size caps, file-size caps, rate limit,
and upstream timeout with those routes in mind. A cheap route can be polled; a COSTLY route cannot.
(§3.3 covers the gateway.)

`build_space_app()` also installs two error handlers:

- a `StarletteHTTPException` handler, and
- a generic `Exception` handler (recorded in the deployment docs as **F-12b**).

The generic handler matters: without it, an unhandled exception would return a framework-default body
that leaks internals. With it, the service returns a translated error. See §8.

### 3.2 The ZeroGPU duration map

The module declares a per-task duration budget used when the service is hosted on a ZeroGPU-style
platform that requires an advance duration declaration:

| Task | Duration |
|---|---|
| `vqa` | 20 |
| `caption` | 20 |
| `grounding` | 45 |
| `change` | 30 |
| `optical_sar` | 45 |
| `change_vqa` | 30 |

The helper `decorate_gpu()` applies the declaration, and `_spaces_module()` resolves the platform
module. The numbers are the declared *budgets*, not measured latencies; the captured grounding run
records a measured `step_001` of 209.873 ms (see the `FRONTEND.md` chapter §7.4), which is a single
step's timing, not a task duration, and the two are not comparable.

`docs/DEPLOYMENT_ARCHITECTURE.md` §3.4 documents this same map as the "ZeroGPU duration map". On the
**active** topology the service runs on a CPU Codespace (`SATQUERY_DEVICE=cpu`, per
`deploy/codespace/launch.sh` and `docs/DEPLOYMENT_TOPOLOGY.md` §5), where the GPU decoration is inert.

### 3.3 The gateway and the `/api/*` mirror

There are two front-facing surfaces, and it is important not to conflate them.

**(a) The gateway (`gateway/app.py`).** A thin front tier that proxies a **4-route allowlist** to the
inference service. Its declarations:

| Symbol | Value | Meaning |
|---|---|---|
| `PROXIED_ROUTES` | 4 routes | The allowlist. |
| `BLOCKED_ROUTES` | empty | Nothing is explicitly blocked. |
| `COSTLY_ROUTES` | `("/v1/analyze", "/v1/assets")` | The routes that cost real work. |

It exposes `/v1/gateway/health` (its own health, distinct from `/v1/health`), installs a
`StarletteHTTPException` handler (F-3), and proxies the four routes. Notable mechanisms inside
`_proxy()`:

- **F-2** — it strips client CORS headers and *asserts* that none remain (`_is_cors_header()`,
  `_CORS_HEADER_PREFIX`). This prevents a client from injecting an `Access-Control-*` header that the
  gateway would then pass upstream.
- **F-6** — it applies a **streaming cap** on the response body rather than buffering unbounded.
- It deliberately **does not retry** (there is an explicit no-retry comment): a retry of a COSTLY route
  would double the work.
- `_read_body_bounded()` (F-9) is a thin adapter that bounds the request body it reads.
- `_client_ip()` derives the client IP (used by the rate limiter), and `_env()` reads configuration.

The gateway is an **optional** front tier. Its module docstring notes it is unimportable in a
sandbox — i.e. it is written to be deployed, not imported by test runners — and the module-level `app`
is created inside a `try/except` for that reason.

**(b) The Render orchestrator (`deploy/render/main.py`, 532 lines).** The orchestrator exposes the
`/api/*` mirror of the four endpoints:

| Orchestrator route | Mirrors |
|---|---|
| `/api/health` | `/v1/health` |
| `/api/infer` | `/v1/analyze` |
| `/api/capabilities` | `/v1/capabilities` |
| `/api/assets` | `/v1/assets` |

This is the surface the frontend actually calls: `SQ.ENDPOINTS` is
`{assets:'/assets', infer:'/infer', capabilities:'/capabilities', health:'/health'}`
(`frontend/assets/js/live.js`) and the default base is `/api`, so the frontend's `/api/infer` maps to
the orchestrator's `/api/infer`, which maps to the service's `/v1/analyze`. Note the name change:
**the frontend says "infer"; the service says "analyze"; they are the same endpoint.**

The orchestrator's internals:

| Symbol | Behaviour |
|---|---|
| `_github_token()` | Reads the GitHub token used to wake the Codespace. |
| `_codespace_name()` | Reads and **strips** the Codespace name — the strip is the fix for the B-02 trailing-`\n` defect (see §12). |
| `_codespace_port()` | Defaults to `8000`. |
| `_wake_timeout_s()` | Defaults to `120`. |
| `_upstream_timeout_s()` | Defaults to `90`. |
| `_DEV_ORIGINS`, `_PRODUCTION_ORIGINS` | `_PRODUCTION_ORIGINS = ("https://satquery.pages.dev",)`; `_allowed_origins()` composes the CORS allowlist. |
| `OrchestratorError`, `WakeTimeout`, `OrchestratorConfigError`, `OrchestratorUpstreamError` | The orchestrator's own error types. |
| `_envelope()` | Wraps a response/error into the orchestrator's envelope shape. |
| `ensure_codespace_up()` | Wakes the Codespace if it is asleep (the wake sequence). |
| `_proxy()` | Forwards the request upstream. |
| `create_app()` | Builds the app with the four routes. |
| `_handle_orchestrator_error()` | Translates an orchestrator error into a response. |

**Documented drift, recorded not hidden.** `deploy/render/main.py`'s own docstring notes that it is
**superseded by the tunnel design** per the delivery documents, while remaining the source present in
this working copy. The deployed backend is the `SatQuery-Backend` repository (`main.py`, 768 lines,
with a tunnel), whose deployed HEAD is `89d80eaddec5` (`release/DOCS_STYLE_GUIDE.md` §3). The local
`deploy/render/main.py` therefore does **not** carry the tunnel implementation. See §9 and §12.

`render.yaml` declares the orchestrator service concretely:

```yaml
startCommand: uvicorn deploy.render.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /api/health
```

with environment variables `PORT`, `SATQUERY_ALLOWED_ORIGINS`, `GITHUB_TOKEN`, `CODESPACE_NAME`,
`CODESPACE_PORT` (`"8000"`), `SATQUERY_DEVICE` (`"cpu"`), `SATQUERY_WAKE_TIMEOUT_S` (`"120"`), and
`SATQUERY_UPSTREAM_TIMEOUT_S` (`"90"`). The plan is free, and **all secret values are declared
`sync: false`** — i.e. they are injected by the platform, not committed. (No value is reproduced in
this chapter; per the release rules, this documentation contains no credentials.)

### 3.4 The `/v1` vs `/api` naming table

Because two naming schemes coexist, here is the mapping in one place:

| Concept | Service (`/v1`) | Orchestrator mirror (`/api`) |
|---|---|---|
| Health | `GET /v1/health` | `GET /api/health` |
| Capabilities | `GET /v1/capabilities` | `GET /api/capabilities` |
| Analysis | `POST /v1/analyze` | `POST /api/infer` |
| Asset upload | `POST /v1/assets` | `POST /api/assets` |
| Gateway's own health | `GET /v1/gateway/health` | — |

The `/v1/` prefix is the service's versioned contract (`docs/API_CONTRACT.md` §1). The `/api/` prefix
is the orchestrator's mirror. A client that speaks `/api/infer` is speaking to the mirror, not to the
service.

---

## 4. The four-endpoint contract in detail

`docs/API_CONTRACT.md` is the frozen, frontend-facing contract (917 lines). This section summarises
what it pins, because the service must satisfy it exactly.

### 4.1 Conventions, and the one schema exception

`docs/API_CONTRACT.md` §1.1: unknown fields are **rejected**. The schemas use Pydantic
`extra="forbid"`, with exactly **one** exception: `GeoMetadata` is `extra="allow"`. The reason is that
geospatial metadata is an open set — a raster may carry CRS, transform, resolution, and arbitrary
derived fields — so forbidding extras there would reject legitimate metadata rather than protect the
contract.

The consequence for a client: sending an unexpected field on any *other* model is a validation error,
not a silently-ignored field. This is a deliberate strictness choice, and it is why the contract is
worth reading before writing a client.

### 4.2 `GET /v1/health` (§2.1)

Returns the health block. Two fields are worth pinning:

- **`device`** is a **closed set**, validated by the F-8 rule in `app/deployment.py`: the legal values
  are `_LEGAL_DEVICES = {cpu, cuda, mps}`. `_effective_device()` returns `None` for an unrecognised
  device rather than echoing it back. So a client can rely on `device` being one of three values or
  absent.
- **`gpu_available: false` is normal on ZeroGPU.** The contract records the *measured* degraded output,
  and states that a `false` here is not a fault on that platform.

The reason to state this in the docs at all: a naive client would treat `gpu_available: false` as an
error. The contract says otherwise.

### 4.3 `GET /v1/capabilities` (§2.2, §2.3, §2.3.1)

Returns the capability block: per-task availability plus a `reason` when a task is unavailable.

- `reason` is **required when `available: false`**. A capability block that said "unavailable" without
  saying why would be less useful than one that names the missing artifact.
- **`modalities` appears only on `optical_sar`.** `app/deployment.py` declares `_MODALITIES` with only
  `optical_sar` in it, so no other task carries a `modalities` field.

**§2.3 / §2.3.1 — the five-word vocabulary, and why `loaded`/`degraded` are never emitted.** The
public capability vocabulary has five states, and `app/deployment.py` translates internal registry
states into them via `CONTRACT_STATES` (5) and `REGISTRY_TO_CONTRACT`. The internal registry states are
`AVAILABLE` / `DEGRADED` / `UNAVAILABLE` (`core/registry.py`), and the registry has
`PLANABLE_STATES` marking which of those the planner may plan against.

The important negative fact: the public contract **never emits the words `loaded` or `degraded`**. The
internal vocabulary and the public vocabulary are deliberately different, and the translation is the
adapter's job. A client that wrote `if status == 'degraded'` would be reading a word the contract does
not use.

### 4.4 `POST /v1/analyze` (§2.4)

Accepts an `AnalysisRequest` and returns a `ResultEnvelope`.

**Multipart is NOT implemented.** This is stated in the contract and it constrains every client: an
asset is uploaded separately to `/v1/assets`, and the analysis request references it by `asset_id`.
A client that tried to send the image inline as a multipart part would be rejected. This is why the
frontend's upload is a raw-bytes POST and why the analysis request is JSON
(`frontend/assets/js/live.js`; `FRONTEND.md` §5.5, §6.8.1).

The request carries the task and, in the serving composition, a `force_task` (see §2.6 — the deployed
controller has no router attached).

The response's fields that the frontend must read are enumerated in the contract: the answer, the
evidence, the regions, the confidence (raw and calibrated), the timings, the provenance (run id,
policy, protocol, schema), the geospatial block, and the warnings. The captured envelope on
`frontend/assets/data/anatomy-run.js` is a real instance of this shape (`FRONTEND.md` §7.4).

**Artifact refs are `null` in v1.** The contract records an explicit ruling (F-16) that artifact
references are `null` — the service does not return a URL or a handle to a produced artifact in v1.
This is a capability limit, not an oversight, and a client must not depend on an artifact ref being
present.

### 4.5 `POST /v1/assets` (§2.5)

Uploads an asset and returns an opaque `asset_id`. The contract records the design as "Option A" and
pins:

| Property | Value |
|---|---|
| `asset_id` opacity | The client must treat the handle as opaque. |
| Size cap | Enforced (F-6 / F-7). |
| Content-type allowlist | Five types. |
| Retries | Documented. |
| Lifetime | The handle is **ephemeral** with a TTL. |

The service-side implementation of all five is in `app/space_app.py` (§7).

### 4.6 Enums (§3)

| Enum | Cardinality | Values |
|---|---|---|
| `Task` | **7** | The task vocabulary. |
| `CoordinateSystem` | **3** | The coordinate-system vocabulary. |
| `Modality` | **4** | The modality vocabulary. |

Seven tasks is worth noting because `core/registry.py`'s `default_specs()` declares **six** specialists
(`vqa`, `caption`, `grounding`, `change`, `change_vqa`, `optical_sar`). The `Task` enum having seven
values while six specialists exist means the enum is the *request* vocabulary and the spec table is the
*implementation* vocabulary; the difference is a task the request enum names but that no specialist
serves directly. Which specific value accounts for the difference:
`UNKNOWN — not established from the available evidence` (the enum's member list was not read
verbatim for this chapter; only its cardinality is recorded here).

### 4.7 The confidence contract (§4)

The contract documents:

- **The measured ECE caveat.** Calibration's ECE went **0.013755 → 0.014929 — worse**. The transform is
  retained only because it is in the frozen config (`release/DOCS_STYLE_GUIDE.md` §3).
- **`T = 0.9772731820958189`** — the temperature.
- **16,441 Val rows** — the calibration sample count. This is the same figure the captured envelope
  records as `calibration_samples: 16441.0` (`frontend/assets/data/anatomy-run.js`; `FRONTEND.md`
  §7.4). The public page and the contract agree.

The honest reading of this section: **the service returns a calibrated confidence, and the calibration
is documented to have made ECE slightly worse.** A client must not present the calibrated confidence as
an accuracy. Per the style guide, there is no end-to-end benchmark, so a per-run confidence is a
per-run confidence.

### 4.8 The error contract (§5)

See §8 for the full treatment. The contract's §5.1 gives the status map, §5.2 the full 23-code
taxonomy, and §5.3 the gateway-origin `rate_limited` code. §5.1 also records the **trailing-slash 307
footgun** (Starlette `redirect_slashes`), which is why a client should compose exact URLs.

### 4.9 Latency, quotas, auth, CORS (§6, §7, §7.1)

- **§6 — latency and quotas.** The contract records the latency expectations and any quotas.
- **§7 — auth: none.** v1 has **no authentication**. This is a first-class design fact with
  consequences that appear all over the codebase: it is why `core/errors.py` scrubs paths (F-15), why
  `app/serving.py` logs rather than attaches the CORS/checkpoint `source`, and why the trace must not
  carry anything sensitive.
- **§7.1 — CORS.** CORS is configured on the orchestrator, whose `_PRODUCTION_ORIGINS` includes the
  Pages origin `https://satquery.pages.dev` (`deploy/render/main.py`). The gateway additionally strips
  client-supplied CORS headers (F-2, `gateway/app.py`).

**§9 — the minimal integration checklist.** The contract closes with a checklist for a new client,
which is the shortest path for anyone writing against this service.

---

## 5. Entrypoint requirements

Any host that runs this service must satisfy five requirements. `docs/DEPLOYMENT_ARCHITECTURE.md`
§3.3 enumerates them, and §3.3.1 adds a sixth consideration (a single capability authority). The
requirements are:

1. **A Python process with the project's dependencies.** `deploy/codespace/launch.sh` performs a
   preflight dependency check for `yaml`, `pydantic`, `fastapi`, `uvicorn`, and `httpx` before it
   starts anything. A host that does not have these cannot start the service.
2. **A callable application object.** `deploy/codespace/serve.py` is the reference implementation:

   ```python
   app = build_space_app()
   uvicorn.run(app, host="0.0.0.0", port=port)
   ```

   with `port = int(os.environ.get("PORT", "8000"))`. The entrypoint therefore must (a) build the app
   via `build_space_app()` and (b) bind a port from the environment with a default.
3. **A port binding on `0.0.0.0`.** The reference binds `0.0.0.0`, not `127.0.0.1`, so the service is
   reachable from outside the process's own namespace.
4. **An environment that can reach the artifacts** (or degrade cleanly without them). Because
   `app/serving.py` wires artifacts through the `builders=` seam and degrades when they are absent, a
   host without the artifacts still *starts* — it just reports the affected capabilities as
   unavailable. This is what makes "degrade, do not crash" a deployment property rather than a slogan.
5. **A health-checkable endpoint.** The orchestrator's `render.yaml` sets
   `healthCheckPath: /api/health`, so the platform probes that path. A host that cannot answer a health
   probe will be considered unhealthy and restarted or removed from rotation.

**§3.3.1 — a single capability authority.** The architecture doc adds that there must be exactly one
authority for capability state: `app/deployment.py`. The registry knows internal state
(`AVAILABLE`/`DEGRADED`/`UNAVAILABLE`); the deployment adapter translates it into the public five-word
vocabulary. A second place that decided capability state would create two answers to "is this task
available?", which is exactly the kind of drift the project's discipline forbids.

### 5.1 The launcher: `deploy/codespace/launch.sh`

`deploy/codespace/launch.sh` is 194 lines and is the reference launcher. Its steps, as read:

1. **Preflight dependency checks** for `yaml`, `pydantic`, `fastapi`, `uvicorn`, `httpx`.
2. **Port and stamp guards** — so two launchers do not fight over the same port and a stale stamp does
   not mislead.
3. **`_restart_serve()`** — starts the service with
   `setsid nohup python deploy/codespace/serve.py`, i.e. detached from the launcher's terminal so the
   service survives the shell.
4. **The supervised tunnel-agent loop** — starts the tunnel agent with
   `setsid nohup bash -c '… python deploy/codespace/tunnel_agent.py …'` and supervises it, restarting
   it if it exits. See §9.
5. **Environment** — exports `SATQUERY_DEVICE=cpu`, `SATQUERY_ASSET_ENABLED=1`,
   `SATQUERY_ASSET_DIR=/tmp/satquery-assets`, and
   `SATQUERY_HUB_URL=https://<backend-host>`.
6. **Verification** — step 3 verifies the agent "announced to hub", so the launcher does not report
   success merely because the process started.

Note that `SATQUERY_DEVICE=cpu` in the launcher matches `SATQUERY_DEVICE: "cpu"` in `render.yaml` and
the CPU-first reconciliation in `docs/DEPLOYMENT_TOPOLOGY.md` §5.

> **Honesty note.** `deploy/codespace/launch.sh` references `deploy/codespace/tunnel_agent.py`, and
> `docs/DEPLOYMENT_TOPOLOGY.md` §2 and `docs/FINAL_DELIVERY_TODO.md` §1.3 both name that file as part
> of the `SatQuery-Inference` deployment. **That file does not exist in this working copy.** The local
> `deploy/` directory is stale/untracked (`docs/FINAL_DELIVERY_REPORT.md` §6 records "local `deploy/`
> stale"; `docs/FINAL_DELIVERY_TODO.md` §5 records the corresponding blocker). What the tunnel agent
> does is therefore described in §9 from the *evidence that does exist* (the launcher's invocation, the
> topology doc's description, and the transport value in the captured envelope), and the agent's
> internals are marked `UNKNOWN — not established from the available evidence`.

---

## 6. Lazy model loading and `cache_max_models: 1`

### 6.1 The lazy-loading contract

The serving tier does **not** load models at import time. Two mechanisms enforce this:

**(a) `build_serving_registry()` constructs nothing.** Its own docstring says it returns "a
`SpecialistRegistry` that constructs nothing yet (`discover()` reads a spec table)." `core/registry.py`
confirms the shape: `default_specs()` returns six spec rows, and `discover()` reads that table. The
spec table is data; no model is instantiated by reading it.

**(b) Builders import lazily.** `_wired_change_builder` imports `build_change_specialist` *inside* the
function, with the stated reason: "so importing `app.serving` stays cheap and does not pull the model
stack (torch) into a process that never serves a change query." The same pattern appears in the other
two builders. The consequence is that `import app.serving` does not import torch at all.

This matters because `core/registry.py`'s `SpecialistRegistry.__init__` has a torch-import path
(`self.device = device or config.device_preference`). Keeping the import inside builders means a
process that only serves, say, capabilities never pays for torch.

### 6.2 The spec table and lazy construction

`core/registry.py`:

| Symbol | Role |
|---|---|
| `RegistryState` | `AVAILABLE` / `DEGRADED` / `UNAVAILABLE`. |
| `PLANABLE_STATES` | Which states the planner may plan against. |
| `SpecialistSpec` | One row of the spec table. |
| `default_specs()` | **Six** rows: `vqa`, `caption`, `grounding`, `change`, `change_vqa`, `optical_sar`. |
| `RegistryEntry` | The registry's record for one spec; `to_trace()` **scrubs `detail`**. |
| `SpecialistRegistry.discover()` | Reads the spec table; constructs nothing. |
| `SpecialistRegistry.available()` | Returns `tuple(sorted(self._specs))` — a sorted tuple, so the order is stable. |
| `SpecialistRegistry.specs()` | The spec table. |
| `SpecialistRegistry.entry(name)` | One entry. |

The `default_specs()` rows carry their asset requirements: `requires_assets` is `1`, `2`, or `None`
depending on the task (a single-image task needs 1; a paired task needs 2; a task that needs no asset
has `None`). They also carry `optional_config_keys`. These are the same requirements the frontend's
`PAIRED_TASKS = {change, change_vqa, optical_sar}` reflects on the client side (`FRONTEND.md` §5.3) —
and it is worth noting the two lists agree: the three paired tasks are exactly the three whose
`requires_assets` is 2.

`RegistryEntry.to_trace()` scrubbing `detail` is a privacy mechanism: the trace reaches the client, and
v1 has no auth, so the entry's raw detail does not travel.

### 6.3 `cache_max_models: 1`

The serving configuration caps the model cache at **one** model. The consequence is the important part:
with a cache of one, serving a task evicts the previously loaded model. A sequence of requests across
two tasks therefore loads and evicts repeatedly rather than holding both.

Why this is the right default for this deployment: the active host is a CPU Codespace
(`SATQUERY_DEVICE=cpu`) with limited memory, and the project's posture is CPU-first
(`docs/DEPLOYMENT_TOPOLOGY.md` §5). Holding several models resident would risk memory exhaustion, and
an `OutOfMemoryError` is a defined failure in the taxonomy (`core/errors.py`, `out_of_memory`,
`recoverable=True`) precisely because memory pressure is an expected condition.

The honest cost of `cache_max_models: 1`: **a multi-task workload pays repeated model-load cost.** This
is a latency property, not a correctness one. It is stated here rather than omitted because it is a
real consequence a reader should know before benchmarking latency.

Where `cache_max_models` is declared in config: `UNKNOWN — not established from the available
evidence` for the exact key location (the value's *effect* — a cap of one — is what is documented
here; the config file line was not read for this chapter).

---

## 7. The asset store

### 7.1 Purpose and shape

`POST /v1/assets` exists because multipart is not implemented (§4.4). An asset is uploaded once,
receives an opaque handle, and the handle is referenced by the analysis request.

`app/space_app.py` implements the store with a module-level cache (`_ASSET_STORE`) and an accessor
`get_asset_store()`. Its configuration comes from environment variables:

| Helper | Default | Meaning |
|---|---|---|
| `_asset_max_files()` | **32** | Maximum number of files held. |
| `_asset_ttl_seconds()` | **900.0** | Handle lifetime, in seconds (15 minutes). |
| `_asset_root()` | system tempdir fallback | Where asset bytes are written. |
| `_asset_max_file_bytes()` | — | Per-file byte cap; **refuses a non-positive or non-integer value** (F-7). |

`_ALLOWED_ASSET_CONTENT_TYPES` declares the **five** accepted content types, matching
`docs/API_CONTRACT.md` §2.5 and the client's `SQ.CONTENT_TYPES` (`frontend/assets/js/live.js`:
`tif`, `tiff`, `png`, `jpg`, `jpeg`).

### 7.2 Fail-closed availability

The store's availability gate is `_asset_store_available()`, which requires **BOTH**:

- `SATQUERY_ASSET_ENABLED`, and
- `SATQUERY_ASSET_DIR`.

If either is missing, the store is unavailable and `POST /v1/assets` returns **503**. This is
**fail-closed**: the service refuses uploads rather than accepting them into a store it cannot
guarantee. That is the correct posture for an ephemeral store — a handle issued by a store that cannot
serve it back is worse than no handle.

The launcher (`deploy/codespace/launch.sh`) sets both:

```
SATQUERY_ASSET_ENABLED=1
SATQUERY_ASSET_DIR=/tmp/satquery-assets
```

so the deployed Codespace has the store enabled with a temp-dir root. On a host where the variables are
absent, the 503 is the expected behaviour and the frontend surfaces it via `translateError()`
(`FRONTEND.md` §6.7).

### 7.3 Handle opacity and lifetime

The handle is `asset_<32 hex>` — 32 hex characters, which is `secrets.token_hex(16)`. Two properties
follow:

1. **It is unguessable.** 16 random bytes (128 bits) means a client cannot enumerate handles.
2. **It is opaque.** Nothing about the underlying file is encoded in it. The client must not parse it,
   and the frontend's `uploadAsset()` explicitly *asserts* the handle exists and passes it back
   unexamined (`frontend/assets/js/live.js`; `FRONTEND.md` §6.8.1).

The **TTL** (default 900.0 s) and the **file cap** (default 32) together mean the store is a short-lived
staging area, not a database. The practical consequences for a client:

- An upload and its analysis must happen **within the TTL**.
- A workload that uploads more than 32 files concurrently will hit the cap.
- Nothing survives a service restart: the store is in-memory plus a temp directory.

### 7.4 Path scrubbing on the way out

`core/errors.py` implements **F-15** path scrubbing (`scrub_paths()`), which is directly relevant to the
asset store because asset errors are client-visible. The mechanism:

- `_WINDOWS_DRIVE_PATH`, `_UNC_PATH`, and `_POSIX_PATH` match **absolute** paths.
- The replacement keeps only the **final component** ("basename reduction"), so
  `"cannot read C:\\a\\b\\weights.pt"` becomes `"cannot read weights.pt"` — "still diagnostic, no
  longer a location disclosure."
- **Relative paths are deliberately not matched**, and the reason is documented: "A rule broad enough to
  catch `artifacts/change/head.pt` also catches `and/or` and the path segments of a URL, and a
  scrubber that mangles ordinary prose is a worse defect than the disclosure it fixes. The measured
  leaks are all absolute."
- **URLs are left intact on purpose**: `https://github.com/antofuller/CROMA` appears inside one of the
  very messages this scrubs, and mangling it "would be a worse defect than the one being repaired."
  The `_POSIX_PATH` lookbehind refuses to start a match immediately after `:` or `/`, which is the
  mechanism that keeps the URL intact.

The module also records the *history* of the fix, which is instructive: a blunt replacement of the
whole message with a generic string was tried first, "but it discarded path-free diagnostics the client
can legitimately act on (`... has no builder 'build_x'`, `no GPU in this dimension`), and three
existing tests that pin exactly those diagnostics failed. **A fix that forces legitimate tests to be
weakened is aimed at the wrong granularity.**"

F-15's owner ruling (2026-09-23) is quoted in the file: *"sanitize all client-facing exception
messages; retain full exception details only in server-side diagnostics."* The reason it was needed:
exception messages in this repo routinely embed an absolute path (e.g. `specialists/optical_sar/croma.py`
raises a message naming a vendored directory; `specialists/change/stanet.py` raises one naming an
encoder-weights path), and those strings reach client-visible fields — and v1 has no auth.

---

## 8. Error translation and machine codes

### 8.1 The taxonomy: 23 codes

`core/errors.py` (316 lines) defines the taxonomy. Every failure the system can produce is one of these
codes, and the module's docstring states the rule plainly: "Never raise a bare Exception from specialist
or controller code."

The base class is `SatQueryError`, whose attributes are documented in the file:

| Attribute | Meaning |
|---|---|
| `code` | Stable machine-readable identifier, used in traces. |
| `user_message` | Text safe to show the operator. |
| `detail` | Technical detail for the execution trace (**never chain-of-thought**). |
| `recoverable` | Whether the controller may continue with a fallback. |

It carries a `to_trace()` method returning `{code, detail, recoverable, context}`.

The taxonomy, grouped as the file groups it:

**Input / raster.**

| Code | Class | `recoverable` |
|---|---|---|
| `input_error` | `InputError` | default |
| `raster_read_error` | `RasterReadError` | default |
| `missing_crs` | `MissingCRSError` | **True** — "Degraded, not fatal: non-geospatial analysis may still be possible." |
| `unsupported_bands` | `UnsupportedBandsError` | default |
| `oversized_image` | `OversizedImageError` | **True** — recoverable via downscale. |

**Pairing.**

| Code | Class | Note |
|---|---|---|
| `pair_incompatible` | `PairCompatibilityError` | — |
| `pair_misaligned` | `PairMisalignmentError` | Subclass of the above. |
| `temporal_pair_invalid` | `TemporalPairError` | Subclass of the above. |

**Routing / planning.**

| Code | Class |
|---|---|
| `routing_error` | `RoutingError` |
| `unsupported_query` | `UnsupportedQueryError` |
| `invalid_request` | `InvalidRequestError` |
| `workflow_plan_error` | `WorkflowPlanError` |

**Specialists.**

| Code | Class | `recoverable` |
|---|---|---|
| `specialist_error` | `SpecialistError` | default |
| `model_load_error` | `ModelLoadError` | default |
| `model_unavailable` | `ModelUnavailableError` | **True** — "the controller degrades the workflow." |
| `out_of_memory` | `OutOfMemoryError` | **True** — retry at lower resolution. |
| `specialist_timeout` | `SpecialistTimeoutError` | **True** |

**Output integrity.**

| Code | Class |
|---|---|
| `schema_validation_error` | `SchemaValidationError` |
| `coordinate_error` | `CoordinateError` |
| `confidence_range_error` | `ConfidenceRangeError` |

**Leakage / evaluation.**

| Code | Class |
|---|---|
| `leakage_violation` | `LeakageError` |
| `benchmark_freeze_error` | `BenchmarkFreezeError` |

That is **23 codes**, matching `__all__`'s 23 entries and the "23-code taxonomy" recorded in
`docs/API_CONTRACT.md` §5.2 and `gateway/policy.py`'s `_CODE_STATUS`.

### 8.2 The `specialist_timeout` recoverability correction

One entry deserves its own treatment because the file documents a *defect* it corrected.
`SpecialistTimeoutError` was inheriting `recoverable=False` from `SatQueryError`, and the file explains
why that was wrong, with two independent reasons:

1. `docs/API_CONTRACT.md` is the frozen frontend-facing contract, and §5.1 **maps 504 with
   `recoverable: true`**. A frontend that reads `recoverable: false` "will not offer a retry for the one
   failure the contract explicitly tells it to retry."
2. The plan's Failure Matrix (§57) lists Timeout with the recovery "abort specialist" and the fallback
   "partial result" — i.e. the controller continues rather than failing the request. A terminal
   `recoverable=False` contradicts that.

The file also records *why the defect was invisible from the inside*: "the controller currently only
reuses `.code` for its budget-skip trace entry (`core/controller.py:464`), so nothing in the pipeline
constructed this class and the wrong default was never observable from the inside — only from a
client." This is a good example of the project's practice of documenting *how* a bug could hide.

### 8.3 The status map and the gateway-origin code

`gateway/policy.py` declares `_CODE_STATUS`, the map from each of the 23 codes to an HTTP status, and:

```python
GATEWAY_ORIGIN_CODES = {"rate_limited"}
_CODE_STATUS["rate_limited"] = 429
```

So `rate_limited` is a **gateway-origin** code: it is not one of the 23 taxonomy codes produced by the
service, it is produced by the gateway's own rate limiter, and it maps to **429**. `docs/API_CONTRACT.md`
§5.3 records it separately for exactly this reason — a client should understand that a 429 came from the
gateway, not from the analysis pipeline.

`DEFECT_CODES` (5) names the codes that indicate a *defect* rather than a normal failure. The
distinction matters: a defect code means the system did something wrong, whereas most codes describe a
legitimate condition (a missing CRS, a bad upload, a timeout).

### 8.4 `translate_error()`

`translate_error()` maps an error to its client-facing form. Its role in the architecture is stated in
`docs/DEPLOYMENT_ARCHITECTURE.md` §2.3: **the code is passed unchanged.** The gateway translates the
*shape* (into its envelope, with a request id) but does not rewrite the code — so a client sees the
service's own code, not a gateway-invented one.

Supporting symbols: `_REQUEST_ID_RE` (validates a request id's shape) and `new_request_id()` (mints
one). A request id is what makes a client-side report correlatable with a server-side log.

### 8.5 `GatewayConfig` and its validators

`gateway/policy.py` declares `GatewayConfig` with these defaults:

| Field | Default |
|---|---|
| `max_body_bytes` | 8 MiB |
| `max_file_bytes` | 4 MiB |
| `rate_limit_per_ip` | 10 |
| `rate_limit_window_s` | 60.0 |
| `upstream_timeout_s` | 90.0 |
| `allowed_content_types` | 5 |

Its `__post_init__` validators reject a misconfiguration rather than letting it fail later:

- an origin with a **trailing slash** is rejected,
- an empty value is rejected,
- a `*` wildcard is rejected,
- and a timeout that is **not greater than 45** is rejected.

The last one is interesting: the 45-second floor is tied to the GPU duration map's longest budget
(`grounding` and `optical_sar` are both **45** in `app/space_app.py`'s `GPU_DURATIONS`). An upstream
timeout below the longest task budget would cut off a legitimate run, so the validator forbids it.

Note the relationship between the two size caps: the gateway's `max_file_bytes` (4 MiB) is *smaller*
than its `max_body_bytes` (8 MiB), which is coherent — a file cap inside a body cap.

### 8.6 The F-12b generic handler

Back in `app/space_app.py`, the generic `Exception` handler (F-12b) is what makes the taxonomy
*airtight at the edge*: an exception that escaped the pipeline's own handling is still translated into a
response rather than surfacing as a framework default. `docs/DEPLOYMENT_ARCHITECTURE.md` §5 lists F-12
and F-12b among the failure modes, alongside F-11, F-13, F-14, F-15, F-15b, F-15c, F-16, F-16c, F-17,
F-18, and F-19. (F-15c is the gateway's transport-failure detail, `_TRANSPORT_FAILURES` /
`_transport_failure_detail()` in `gateway/app.py`.)

---

## 9. The tunnel agent and the transport

### 9.1 Why a tunnel exists

The service runs on a host (a GitHub Codespace) that is not directly reachable at a stable public
address in the way a normal web service is. The orchestrator on Render is the public face. Something
must carry a request from the orchestrator to the service. That "something" is the transport, and the
captured envelope records the transport it used:

```
transport: "tunnel"
```

(`frontend/assets/data/anatomy-run.js`; `FRONTEND.md` §7.4). The frontend's live client also reads a
transport response header, `x-satquery-transport` (`frontend/assets/js/live.js`), which is how a client
can see which transport carried its response.

### 9.2 The two transports

`docs/DEPLOYMENT_TOPOLOGY.md` and the delivery documents describe two transport designs:

1. **Forwarded-port transport.** The orchestrator reaches the Codespace through a forwarded port. In
   this design a private repository yields a **302** (a redirect), which is why a 302 is a documented
   behaviour rather than an error.
2. **Outbound tunnel transport.** The service-side agent **long-polls** `POST /tunnel/agent` to the
   hub, so the connection is *outbound* from the Codespace. An outbound tunnel avoids requiring the
   Codespace to be reachable inbound, which is the property that makes it robust on a platform that
   does not expose inbound ports.

The tunnel design supersedes the forwarded-port design: `deploy/render/main.py`'s docstring says it is
superseded by the tunnel design per the delivery documents, and the deployed backend repository is the
one that carries the tunnel.

### 9.3 The agent's role, and what is known about it

The agent's role, assembled from the evidence that exists:

- **`deploy/codespace/launch.sh` starts and supervises it** with
  `setsid nohup bash -c '… python deploy/codespace/tunnel_agent.py …'`, detached from the launcher's
  terminal and restarted if it exits. So the agent is a long-running process, not a one-shot.
- **It announces to the hub.** The launcher's step 3 verifies that the agent "announced to hub", so
  announcing is part of the agent's contract and the launcher treats a failed announcement as a failed
  launch.
- **`SATQUERY_HUB_URL` names the hub.** The launcher sets it to
  `https://<backend-host>`, which is the same host the frontend's
  `<meta name="satquery-api-base">` names (`frontend/mission.html`). So the hub, the orchestrator, and
  the API base are one host.
- **It is supervised, and it is started after the service.** The launcher starts the service
  (`_restart_serve()`) and *then* starts the agent, which is the correct order: an agent that
  announced before the service was listening would advertise a dead endpoint.

**What the agent does internally** — its poll loop, its request framing, its reconnection strategy, its
handling of a hub restart — is `UNKNOWN — not established from the available evidence`, because
`deploy/codespace/tunnel_agent.py` does not exist in this working copy (§5.1's honesty note). The
deployed backend repository (HEAD `89d80eaddec5`) is where the tunnel implementation lives, and it was
not read for this chapter.

### 9.4 B-07: tunnel gaps, patch prepared but not deployed

Per `release/DOCS_STYLE_GUIDE.md` §3 and `docs/FINAL_DELIVERY_TODO.md` §5: **B-07 is OPEN. It is tunnel
gaps, and the patch is prepared but NOT deployed.** This status must not be upgraded. The correct
statement is:

> B-07 — tunnel gaps. Patch prepared, not deployed. **OPEN.**

The consequence for a reader: the tunnel transport works well enough to have carried the runs recorded
in the delivery documents (including the captured `run_d124d8b9adea`, whose `transport` is `"tunnel"`),
and it also has known gaps whose fix is written but not live. Both halves are true at once.

---

## 10. The deployment topology

### 10.1 The active topology

`docs/DEPLOYMENT_TOPOLOGY.md` is the **active** topology document. Its components:

| Component | Host | Role |
|---|---|---|
| Static tier | Cloudflare Pages | The eleven pages (see `FRONTEND.md`). |
| Public backend | Render (`satquery-orchestrator`) | The `/api/*` mirror; wake + proxy; CORS. |
| Inference | GitHub Codespace | Runs the service (`build_space_app()`), CPU-first, plus the tunnel agent. |
| Model artifacts | Hugging Face | Artifact hosting; also the public release surface. |

The document contains a Mermaid topology diagram and a **wake sequence**, plus §3's per-component
responsibilities and environment variables, §4's five old blockers, §5's reconciliation (CPU-first),
and §6's preconditions.

### 10.2 Deployed HEADs

Per `release/DOCS_STYLE_GUIDE.md` §3:

| Component | Deployed HEAD |
|---|---|
| Frontend | `2d7ae53b482d` |
| Backend | `89d80eaddec5` |
| Inference | `5a0936ace491` |

### 10.3 The measured live environment

`docs/DEPLOYMENT_TOPOLOGY.md` §3.2 records the **measured live env-var set**. Two entries in that
section are worth flagging because the section also notes that some names listed historically are
**not** in the live config: `SATQUERY_UPSTREAM_URL` and `HF_TOKEN` are named in the section's own prose
while the section's measured note says they are not present. This is documentation drift inside the
topology document, recorded here rather than propagated.

`docs/DEPLOYMENT_ARCHITECTURE.md` carries a superseded-topology banner and still names Railway /
HF-Space hosts in its body while the active hosts are Render / Codespace. Both documents are kept, with
the banner making the supersession explicit — which is the project's stated practice (mirroring
`P10-T02`).

### 10.4 `docs/DEPLOYMENT_ARCHITECTURE.md` §2 — gateway responsibilities

The architecture document's §2 enumerates the gateway's responsibilities and the 4-route allowlist, and
§2.3 pins the error-translation rule (code passed unchanged). §3.1 assigns entrypoint ownership, §3.2
lists constraints, §3.3 lists the five entrypoint requirements, §3.3.1 the single capability authority,
§3.4 the ZeroGPU duration map, §4 the env-var vocabulary (a long table with F-6/F-7/F-8/F-9 notes), §5
the failure-mode table (F-11…F-19), §6 what is excluded, §7 implementation status, and §8 deployment
preconditions.

### 10.5 The pipeline the service runs

The service's work is done by `core/controller.py`'s `AnalysisController.run()`, whose stages are:

```
RECEIVE → PARSE → VALIDATE → PLAN → EXECUTE → AGGREGATE → VERIFY → RESPOND
```

The captured grounding envelope's eight steps are `RECEIVE` → `RESPOND`, i.e. the same eight-stage
pipeline (`frontend/assets/data/anatomy-run.js`). Notable details from `core/controller.py`:

- **`_asset_label`** — a basename reduction applied to asset labels, the same idiom as F-13/F-14 and
  the same idiom `core/errors.py::scrub_paths` uses for F-15. "One rule, one implementation, applied at
  every client-facing write site."
- **F-19** — the registry is **re-snapshotted after execute**:
  `trace.parameters["registry"] = self.registry.describe()` is written *after* the EXECUTE stage, so the
  trace records the registry state that actually ran rather than the state at request entry.
- **`_execute()`** — applies a **budget between steps**; and per F-15, sets
  `trace.errors[].message = user_message` (the sanitized message, not the raw detail).
- **`_execute_one()`** — implements **F-20**, a producer-side repair for unhandled exceptions, so a
  specialist that raises something unexpected is still recorded as a result rather than escaping.
- **`health()`** — **deprecated**: it "Constructs everything", and it was retired as the public path.
  This is why `app/deployment.py` owns the health payload instead: the public health path must be
  cheap, and a health check that constructs every model is not cheap.
- `_route()`, `_resolve_assets()`, `_modalities()` — the routing, asset-resolution, and modality
  helpers.

---

## 11. What the service does NOT do

Stated explicitly, because the depth of §2–§10 could otherwise imply more capability than exists.

- **No Gradio GUI.** The service is an HTTP API. There is no Gradio interface in this serving tier; the
  user interface is the static frontend (`FRONTEND.md`), which talks to the service over HTTP. Whether
  a Gradio surface exists anywhere else in the project: `UNKNOWN — not established from the available
  evidence` for this chapter (the serving modules read contain no Gradio application).
- **No streaming.** There is no server-sent-events or websocket channel. A request is answered with a
  single response. The frontend's eight-event display is driven *client-side* from that one response
  plus two headers (`X-SatQuery-State`, `x-satquery-transport`), not pushed from the server
  (`FRONTEND.md` §14).
- **No batching.** A request is one analysis. There is no batch endpoint, and `POST /v1/analyze` takes
  one `AnalysisRequest`.
- **No queue.** There is no job queue and no async job model: a COSTLY route does its work within the
  request, bounded by the upstream timeout (`upstream_timeout_s` default 90.0) and the gateway's
  timeout floor (> 45). This is why the gateway deliberately does **not** retry (`gateway/app.py`): a
  retry of a COSTLY route would duplicate work rather than dequeue it.
- **No authentication.** v1 has no auth (`docs/API_CONTRACT.md` §7). This has downstream consequences
  throughout: path scrubbing (F-15), trace scrubbing (`RegistryEntry.to_trace()` scrubs `detail`), and
  logging-instead-of-attaching composition facts (`app/serving.py`).
- **No multipart upload.** Assets are uploaded separately (`docs/API_CONTRACT.md` §2.4).
- **No artifact refs.** Artifact references are `null` in v1 (the F-16 ruling).
- **No persistence.** The asset store is ephemeral (TTL 900.0 s, cap 32 files) and there is no run
  store. A restart loses everything.
- **No natural-language routing in the serving composition.** `build_serving_controller()` attaches no
  router, so a caller drives it with `force_task` (`app/serving.py`; §2.6).
- **No model preloading.** Models load lazily and the cache holds one (`cache_max_models: 1`; §6).
- **No end-to-end benchmark.** Per `release/DOCS_STYLE_GUIDE.md` §3 this does not exist, and no
  system-level accuracy is claimed anywhere in this chapter.

---

## 12. Status summary and blockers

### 12.1 Status by subsystem

| Subsystem | Status |
|---|---|
| `app/serving.py` composition root (`build_serving_registry`, `build_serving_controller`) | `IMPLEMENTED` |
| Artifact wiring via the `builders=` seam (change / change_vqa / optical_sar) | `IMPLEMENTED` |
| `app/space_app.py` (`build_space_app()`, four routes, two error handlers) | `IMPLEMENTED` |
| `app/deployment.py` capability adapter (two vocabularies, five contract states) | `IMPLEMENTED` |
| Four-endpoint contract (`/v1/health`, `/v1/capabilities`, `/v1/analyze`, `/v1/assets`) | `IMPLEMENTED` |
| `/api/*` orchestrator mirror | `IMPLEMENTED`; deployed backend HEAD `89d80eaddec5` |
| Gateway (4-route allowlist, `COSTLY_ROUTES`, F-2/F-3/F-6/F-9) | `IMPLEMENTED` |
| Lazy model loading; `cache_max_models: 1` | `IMPLEMENTED` |
| Asset store (opaque handles, TTL, cap, allowlist, fail-closed 503) | `IMPLEMENTED` |
| Error taxonomy (23 codes) + `_CODE_STATUS` + gateway-origin `rate_limited` | `IMPLEMENTED` |
| Path scrubbing (F-15) | `IMPLEMENTED` |
| Tunnel transport | `IMPLEMENTED`; carried `run_d124d8b9adea` (`transport: "tunnel"`) |
| B-02 `codespace_name` trailing `\n` | Fixed in `deploy/render/main.py` via a strip; recorded as cosmetic, **OPEN** |
| B-07 tunnel gaps | Patch prepared, **NOT deployed** — **OPEN** |

### 12.2 The blockers, stated exactly

| ID | Statement | Status |
|---|---|---|
| **B-07** | Tunnel gaps. Patch prepared, not deployed. | **OPEN** — never to be upgraded. |
| **B-02** | `codespace_name` trailing `\n`. Cosmetic. The orchestrator's `_codespace_name()` strips it. | **OPEN** (cosmetic) |
| Local `deploy/` | The local `deploy/` directory is stale/untracked; `deploy/codespace/tunnel_agent.py` is absent; `deploy/render/main.py` is superseded by the deployed backend. | **KNOWN** (`docs/FINAL_DELIVERY_TODO.md` §5 B-03; `docs/FINAL_DELIVERY_REPORT.md` §6) |
| Change capability | Recorded as degraded in the delivery documents at the time of writing. | `KNOWN` — per `docs/FINAL_DELIVERY_REPORT.md` §6 |
| P2-T03 | Cosmetic. | **OPEN** (cosmetic) |

`docs/FINAL_DELIVERY_TODO.md` §5 records the full blocker register: B-01 **CLOSED**, B-02
**DOWNGRADED**, B-03 **KNOWN**, B-04 **ACCEPTED**, B-05 **ACCEPTED**, B-06 **KNOWN**, B-07 **OPEN**,
B-08 **CLOSED**. Note that B-01 (which `docs/FINAL_DELIVERY_REPORT.md` §6 records as HF BLOCKED at the
time of that report) is **CLOSED** in the later TODO register — so the correct current statement is
that B-01 is CLOSED, with the earlier report's BLOCKED status being superseded.

### 12.3 The G-1 annotation-scope defect

This is the most instructive serving defect in the project and deserves its own treatment.

**The mechanism.** `app/space_app.py` uses `from __future__ import annotations`. Under that import,
annotations are **strings**, resolved lazily by FastAPI via `eval` against a namespace. If a parameter's
annotation names a type (`Request`) that is **bound in a narrower scope** than the function that FastAPI
introspects, then FastAPI's `eval` resolves that name against the **wrong globals**. The name fails to
resolve as a type, and FastAPI **silently reinterprets the parameter as a REQUIRED QUERY PARAMETER named
`request`**.

**The symptom.** Every upload gets:

```json
422 {"detail":[{"loc":["query","request"]}]}
```

This is the worst kind of bug: a **server-side** defect that presents as a **client-side** validation
error. A client developer reads "missing required query parameter `request`" and concludes they
mis-called the API. They did not.

**Why it is silent.** There is no exception at import time. The app builds. The route registers. Only
the *interpretation* of the parameter changed, and it changed in a way that produces a plausible-looking
error.

**The twin, and the asymmetry.** The related case is a **return annotation** naming `JSONResponse`. In
that case the resolution failure does **not** degrade silently — it raises **`PydanticUndefinedAnnotation`**,
and it raises **at import/definition time**, so `build_space_app()` is **never called at all**. The app
therefore does not exist.

So the defect has two halves with **opposite** failure modes:

| Annotation position | Failure mode |
|---|---|
| **Parameter** annotation | **Silent.** The parameter is reinterpreted as a required query parameter. The app runs and every upload 422s. |
| **Return** annotation | **Loud.** `PydanticUndefinedAnnotation` is raised before `build_space_app()` can be called; the app never starts. |

The asymmetry is why the defect is worth documenting: the *loud* half is easy to find (the app will not
start), and the *silent* half is the dangerous one (the app starts and lies about why it is failing).

**The repair pattern.** `app/space_app.py` lines 55–91 carry module-scope comment blocks binding
`Request`, `Response`, and `JSONResponse` at **module scope**, so that FastAPI's `eval` resolves the
names against the module's globals. The gateway has the **twin** of this: `gateway/app.py` also binds
`Request`, `Response`, and `JSONResponse` at module level for the same reason. The rule extracted:

> **Under `from __future__ import annotations`, every type used in a FastAPI route signature must be
> bound at the module scope where the route function is defined — because FastAPI resolves annotations
> by `eval` against that module's globals, and a narrower-scope binding resolves to nothing.**

The correct status for G-1: the **repair is IMPLEMENTED** (the module-scope bindings are present in both
`app/space_app.py` and `gateway/app.py`). The **defect is RESOLVED** in the code read. Whether an
earlier deployment ever served the silent-422 behaviour is a historical question: the recorded live
validation ran 24 runs with 8/8 per pass (`release/DOCS_STYLE_GUIDE.md` §3), which is consistent with a
working upload path in the deployed build — but the exact deployment at which the fix landed is
`UNKNOWN — not established from the available evidence`.

### 12.4 Other failure modes recorded in the architecture doc

`docs/DEPLOYMENT_ARCHITECTURE.md` §5 lists the failure-mode table. The ones most relevant to serving:

| ID | Subject |
|---|---|
| F-6 | Streaming size cap (also `gateway/app.py` `_proxy()`). |
| F-7 | `_asset_max_file_bytes()` refuses a non-positive or non-integer value. |
| F-8 | `device` validation → `_effective_device()` returns `None` for an unrecognised value; `_LEGAL_DEVICES = {cpu, cuda, mps}`. |
| F-9 | `_read_body_bounded()` in the gateway. |
| F-11 | (per §5) |
| F-12 / F-12b | The generic exception handler in `build_space_app()`. |
| F-13 / F-14 | `_asset_label` basename reduction. |
| F-15 / F-15b | Path scrubbing; the F-15b variant. |
| F-15c | Gateway transport-failure detail (`_TRANSPORT_FAILURES`, `_transport_failure_detail()`). |
| F-16 / F-16c | The artifact-refs-`null` ruling; the F-16c variant. |
| F-17 / F-18 / F-19 | F-19 is the post-execute registry re-snapshot in `core/controller.py`. |
| F-20 | Producer-side repair for unhandled exceptions in `_execute_one()`. |

`docs/DEPLOYMENT_ARCHITECTURE.md` §4's env-var vocabulary table carries the F-6/F-7/F-8/F-9 notes
inline, and §6 states what is excluded from the deployment, §7 its implementation status, and §8 the
deployment preconditions.

---

## 13. NOT RUN / OPEN / BLOCKED (serving)

Per `release/DOCS_STYLE_GUIDE.md` §4, every doc ends with this list.

**NOT RUN**
- No end-to-end benchmark of the service (project-wide fact per `release/DOCS_STYLE_GUIDE.md` §3; the
  service is not exempt, and no system-level accuracy is claimed).
- No load/latency benchmark of the four endpoints under `cache_max_models: 1`.
- No test of the tunnel under a hub restart.
- No verification of the gateway's rate limiter under sustained load.
- No verification of the asset store's cap (32) and TTL (900.0 s) boundaries end to end.

**OPEN**
- **B-07 — tunnel gaps. Patch prepared, NOT deployed.** OPEN. (Never to be upgraded.)
- **B-02 — `codespace_name` trailing `\n`. Cosmetic.** OPEN. (The strip is present in
  `deploy/render/main.py`.)
- **P2-T03 — cosmetic.** OPEN.
- **F-15 path scrubbing** — the *measured* leaks are all absolute paths; relative-path leaks were
  deliberately not covered. The scoping is documented as intentional; whether any relative-path leak
  exists is `UNKNOWN — not established from the available evidence`.
- **Documentation drift inside the topology docs** — `docs/DEPLOYMENT_TOPOLOGY.md` §3.2 names
  `SATQUERY_UPSTREAM_URL` and `HF_TOKEN` while its own measured note says they are not in the live
  config; `docs/DEPLOYMENT_ARCHITECTURE.md` names Railway / HF-Space hosts under a superseded-topology
  banner. Recorded; OPEN as documentation debt.
- **`deploy/codespace/tunnel_agent.py`** — referenced by `launch.sh` and two delivery docs, absent from
  this working copy. The agent's internals are
  `UNKNOWN — not established from the available evidence`.
- **`Task` enum's seventh value** — the enum has seven values while six specialists are declared; which
  value accounts for the difference is `UNKNOWN — not established from the available evidence`.
- **`cache_max_models` config key location** — the value's effect (a cap of one) is documented; the
  exact key location is `UNKNOWN — not established from the available evidence`.
- **G-1's fix deployment point** — the repair is IMPLEMENTED in the code read; the deployment at which
  it landed is `UNKNOWN — not established from the available evidence`.
- **B-01** — `CLOSED` per `docs/FINAL_DELIVERY_TODO.md` §5 (superseding the earlier report's BLOCKED
  status). Recorded here so it is not re-opened.
- **No LICENSE file exists** — project-wide, OPEN (`release/DOCS_STYLE_GUIDE.md` §3).

**BLOCKED**
- Nothing in the serving *code* read for this chapter is blocked.
- **Deployment-level:** the local `deploy/` tree is stale/untracked, so the tunnel implementation
  cannot be read from this working copy — the corresponding investigation is BLOCKED on that tree being
  refreshed (or on the deployed backend repository being read instead).
- **B-01 at the time of `docs/FINAL_DELIVERY_REPORT.md`** was BLOCKED (HF); it is CLOSED per the later
  TODO register. The earlier status is superseded, not deleted.

---

## 14. Where the evidence lives

| Claim area | Evidence file(s) |
|---|---|
| Composition root; the three artifact constants and their identities; the `builders=` seam and why config must not be edited; degrade-don't-crash; the three builders and the defects they close; `build_serving_registry()`; `build_serving_controller()` (no router → `force_task`) | `app/serving.py` |
| HTTP application; `build_space_app()`; the four routes; the two error handlers (incl. F-12b); `GPU_DURATIONS`; `decorate_gpu()`; `_spaces_module()`; `get_controller()`; `describe_deployment()`; the asset-store helpers (`_asset_max_files()` 32, `_asset_ttl_seconds()` 900.0, `_asset_root()`, `_asset_max_file_bytes()` F-7, `_ALLOWED_ASSET_CONTENT_TYPES` 5, `_asset_store_available()` requiring both env vars); `main()` | `app/space_app.py` |
| Capability adapter: `CONTRACT_STATES` (5), `REGISTRY_TO_CONTRACT`, `_REQUIREMENTS`, `_MISSING_REASONS`, `_HUB_REASONS`, `_optical_sar_artifacts()`, `_resolve_croma_checkpoint()`, `_requirement_artifacts()`, `_missing_shipped()`, `_hub_unconfigured()`, `_configured_path()`, `_HUB_BACKED`, `CapabilityReport`, `_MODALITIES`, `DeploymentReport`, `_schema_version()`, `_registry_capabilities()`, `_asset_count()`, `_artifact_evidence()`, `_report_for()`, `deployment_report()`, `_effective_device()` (F-8), `_LEGAL_DEVICES`, `_cuda_detected()`, `health_payload()`, `capabilities_payload()` | `app/deployment.py` |
| Error taxonomy (23 codes), `SatQueryError` + `to_trace()`, the `specialist_timeout` recoverability correction, F-15 path scrubbing (`_WINDOWS_DRIVE_PATH`, `_UNC_PATH`, `_POSIX_PATH`, `scrub_paths()`) | `core/errors.py` |
| `_CODE_STATUS` (23 codes), `DEFECT_CODES` (5), `GATEWAY_ORIGIN_CODES`, `rate_limited` → 429, `translate_error()`, `_REQUEST_ID_RE`, `new_request_id()`, `GatewayConfig` + validators | `gateway/policy.py` |
| Gateway: `PROXIED_ROUTES` (4), `BLOCKED_ROUTES`, `COSTLY_ROUTES`, `/v1/gateway/health`, F-3 handler, `_read_body_bounded()` (F-9), `_proxy()` (F-2 CORS strip + assertion, F-6 streaming cap, no-retry), `_is_cors_header()`, `_CORS_HEADER_PREFIX`, `_client_ip()`, `_env()`, module-level `Request`/`Response`/`JSONResponse` bindings (the G-1 twin) | `gateway/app.py` |
| Registry: `RegistryState`, `PLANABLE_STATES`, `SpecialistSpec`, `default_specs()` (6 rows, `requires_assets`), `RegistryEntry.to_trace()` scrubs `detail`, `discover()`, `available()`, `specs()`, `entry()`; the `builders=` override site (lines 420-433) and the spec-name key (lines 204-213); `_builder_kwargs` (lines 435-453) | `core/registry.py` |
| Controller: the eight-stage pipeline; `_asset_label` (F-13/F-14); the F-19 post-execute registry re-snapshot; `health()` deprecated ("Constructs everything"); `_route()`, `_resolve_assets()`, `_modalities()`, `_execute()` (budget; F-15 `user_message`), `_execute_one()` (F-20) | `core/controller.py` |
| The frozen contract: conventions + `extra="forbid"` / `GeoMetadata extra="allow"` (§1.1); health (§2.1, device closed set, `gpu_available: false` normal on ZeroGPU); capabilities (§2.2, §2.3, §2.3.1 five-word vocabulary, `modalities` only on optical_sar); analyze (§2.4, multipart NOT implemented, artifact refs `null` per F-16); assets (§2.5, opacity, caps, allowlist, lifetime); enums (§3, Task 7 / CoordinateSystem 3 / Modality 4); confidence (§4, ECE 0.013755→0.014929, T = 0.9772731820958189, 16,441 Val rows); errors (§5, §5.1 status map + 307 footgun, §5.2 23 codes, §5.3 `rate_limited`); latency/quotas (§6); auth (§7) + CORS (§7.1); status (§8); integration checklist (§9) | `docs/API_CONTRACT.md` |
| Five entrypoint requirements; §3.3.1 single capability authority; gateway responsibilities + 4-route allowlist + COSTLY; §2.3 error translation (code unchanged); §3.4 ZeroGPU duration map; §4 env-var vocabulary; §5 failure-mode table (F-11…F-19); §6 exclusions; §7 status; §8 preconditions; superseded-topology banner | `docs/DEPLOYMENT_ARCHITECTURE.md` |
| Active topology; components; Mermaid topology + wake sequence; §3 per-component responsibilities/env vars; §4 five old blockers; §5 CPU-first reconciliation; §6 preconditions | `docs/DEPLOYMENT_TOPOLOGY.md` |
| Orchestrator: `_github_token()`, `_codespace_name()` (strip = B-02), `_codespace_port()` 8000, `_wake_timeout_s()` 120, `_upstream_timeout_s()` 90, `_DEV_ORIGINS`, `_PRODUCTION_ORIGINS`, `_allowed_origins()`, error classes, `_envelope()`, `ensure_codespace_up()`, `_proxy()`, `create_app()` (four `/api/*` routes), `_handle_orchestrator_error()`; the superseded-by-tunnel docstring | `deploy/render/main.py` |
| Orchestrator service declaration: start command, `healthCheckPath: /api/health`, env-var names, `sync: false` on secrets | `render.yaml` |
| Service entrypoint: `app = build_space_app()`; `uvicorn.run(host="0.0.0.0", port=...)`; `PORT` default 8000 | `deploy/codespace/serve.py` |
| Launcher: preflight deps; port/stamp guards; `_restart_serve()`; the supervised tunnel-agent loop; the env vars (`SATQUERY_DEVICE=cpu`, `SATQUERY_ASSET_ENABLED=1`, `SATQUERY_ASSET_DIR`, `SATQUERY_HUB_URL`); the "announced to hub" verification | `deploy/codespace/launch.sh` |
| Captured run: `run_id`, `transport: "tunnel"`, `config_hash`, confidence + `temperature` + `calibration_samples`, warnings, steps | `frontend/assets/data/anatomy-run.js` |
| Deployed HEADs (`2d7ae53b482d`, `89d80eaddec5`, `5a0936ace491`); B-07 OPEN patch prepared not deployed; B-02 cosmetic OPEN; no E2E benchmark; live validation 24 runs / 0 mock nodes / 94.4444 % | `release/DOCS_STYLE_GUIDE.md` |
| Commits; live topology; E2E run-id table; metrics; blockers; test results (94 + 183 passed); truthfulness statement | `docs/FINAL_DELIVERY_REPORT.md` |
| Status board; artifact inventory; real measured metrics; nine known blockers (incl. item 9 Cloudflare concatenation); blocker register B-01…B-08; evidence register E-01…E-14; final verification checklist | `docs/FINAL_DELIVERY_TODO.md` |
