# SatQuery AI — Render Orchestrator

The thin, stateless API gateway that sits between the **Cloudflare Pages
frontend** and the **GitHub Codespace inference server**. It holds no model, no
state, no database, and performs **no authentication** (per plan §73/§74 and
`docs/DEPLOYMENT_ARCHITECTURE.md` §1.1).

```
Cloudflare Pages (frontend)
    │  HTTPS, JSON
    ▼
Render — this service   [served by uvicorn on $PORT]
    │  server-to-server
    ▼
GitHub Codespace (inference)   [public forwarded port, CODESPACE_PORT]
    │  GET /v1/health · GET /v1/capabilities
    │  POST /v1/analyze · POST /v1/assets
    ▼
SatQuery serving + controller  (app/space_app.py)
```

## Files

| File | Purpose |
|---|---|
| `deploy/render/main.py` | FastAPI orchestrator (`create_app()` → `app`). |
| `deploy/render/codespaces.py` | GitHub Codespaces REST client (get / start / forwarded-URL). |
| `deploy/render/requirements.txt` | Python deps. |
| `render.yaml` | Render blueprint (web service). |

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `10000` | Supplied by Render. uvicorn binds here. |
| `SATQUERY_ALLOWED_ORIGINS` | `""` | Comma-separated CORS allowlist (frontend origin). Never `*`. |
| `GITHUB_TOKEN` | — | PAT with the `codespace` scope. Used to wake a stopped Codespace. |
| `CODESPACE_NAME` | — | Name of the Codespace running the inference server. |
| `CODESPACE_PORT` | `8000` | Port the inference server listens on (separate from `PORT`). |
| `SATQUERY_WAKE_TIMEOUT_S` | `120` | Max seconds to wait for `/v1/health` after starting the Codespace. |
| `SATQUERY_UPSTREAM_TIMEOUT_S` | `90` | Timeout for the proxy call to `/v1/analyze`. |
| `SATQUERY_DEVICE` | — | Passed through for visibility only; the orchestrator does not load models. |

`PORT`, `GITHUB_TOKEN`, `SATQUERY_ALLOWED_ORIGINS`, and `CODESPACE_NAME` are
`sync: false` in `render.yaml` and must be set in the Render dashboard.

## The wake flow

`POST /api/infer` is the main path and is **blocking wake-then-proxy**:

1. `ensure_codespace_up()` calls the GitHub API for the Codespace.
2. If `state != "available"`, it `POST .../start`s it.
3. It polls `GET {codespace}/v1/health` until `200` or until
   `SATQUERY_WAKE_TIMEOUT_S` elapses.
4. On success it proxies `POST {codespace}/v1/analyze` with the verbatim body and
   returns the upstream JSON.

If the engine cannot be woken in time, the orchestrator returns `504` with
`recoverable: true` (the frontend may retry). The response carries
`X-SatQuery-State: waking` when a cold start occurred, `ready` otherwise — the
frontend shows "Waking inference engine..." on slow responses independently.

The other routes proxy the same way: `GET /api/capabilities` →
`/v1/capabilities`, `POST /api/assets` → `/v1/assets`. **Capabilities are never
answered locally** — there is deliberately no second copy of the capability
table; the gateway is a proxy.

## The CORS fix

The orchestrator registers Starlette's `CORSMiddleware` with
`allow_origins=<SATQUERY_ALLOWED_ORIGINS split on ",">`, `allow_methods=["*"]`,
`allow_headers=["*"]`, `allow_credentials=False`. `CORSMiddleware` answers the
`OPTIONS` preflight itself, which resolves the earlier `405` on preflight.

## Error translation

Upstream failures are wrapped in the v1 envelope
(`{"error": {"code", "message", "detail", "recoverable"}}`, see
`docs/DEPLOYMENT_ARCHITECTURE.md` §2.3):

- Codespace unreachable (connection error) → `502`, `recoverable: true`.
- Wake timed out → `504`, `recoverable: true`.
- Non-JSON upstream body → `502`, `recoverable: true`.
- Misconfigured orchestrator (missing `GITHUB_TOKEN` / `CODESPACE_NAME`) → `500`.

Codes are **not** invented or remapped; the gateway passes through the Space's
own error codes.

## Running locally

```bash
PORT=10000 \
GITHUB_TOKEN=ghp_xxx \
CODESPACE_NAME=my-codespace \
SATQUERY_ALLOWED_ORIGINS="https://satquery.pages.dev" \
uvicorn deploy.render.main:app --port 10000
```

Then exercise it:

```bash
curl http://localhost:10000/api/health
curl -X POST http://localhost:10000/api/infer -H 'content-type: application/json' -d '{"query":"..."}'
curl http://localhost:10000/api/capabilities
```

## Assumptions

- The Codespace's public forwarded URL follows
  `https://{name}-{port}.app.github.dev`. `forwarded_url()` prefers the
  `web_url` returned by the GitHub API (rewriting its port) and falls back to the
  name-based form. **This host pattern was not verifiable from the build
  environment** (no live Codespace to inspect) and should be confirmed against a
  running Codespace.

Nothing here is claimed as deployed.
